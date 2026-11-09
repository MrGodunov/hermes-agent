import sqlite3

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def append(db, *, epoch, sequence, content, **assignment):
    return rt.mutate_worker_execution(db, epoch=epoch, **assignment, sequence=sequence,
        operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': content}]})


@pytest.mark.parametrize('recover_first', [False, True])
def test_explicit_worker_adoption_preserves_claim_without_reexecution(tmp_path, recover_first):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        accepted = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'work'})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assignment = dict(execution_id='compute', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
        new_epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        if recover_first:
            rt.recover_session_inputs(db, epoch=new_epoch)
        rt.adopt_worker_execution(db, epoch=new_epoch, **assignment, adoption_secret='private')
        rt.recover_session_inputs(db, epoch=new_epoch)
        restored = rt.get_session_admission(db, admission_id=accepted['admission_id'])
        assert restored['status'] == 'started' and restored['owner_epoch'] == new_epoch
        assert restored['generation'] == claim['generation']
        assert rt.claim_session_input(db, epoch=new_epoch, session_id='s') is None
        message_args = dict(epoch=new_epoch, **assignment, sequence=1, content='committed')
        receipt = append(db, **message_args)
        result = rt.settle_session_input(db, epoch=new_epoch, admission_id=accepted['admission_id'], generation=claim['generation'], outcome='completed')
        assert result['status'] == 'terminal'
        before = db._conn.total_changes
        assert append(db, **message_args) == receipt
        assert db._conn.total_changes == before
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            append(db, epoch=new_epoch, **assignment, sequence=2, content='late')
    finally:
        db.close()


@pytest.mark.parametrize('refuse_revoke', [False, True])
def test_unknown_resolution_atomically_revokes_worker_before_unblocking(tmp_path, refuse_revoke):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        first = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='first', payload={})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assignment = dict(execution_id='worker', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
        lease = dict(epoch=epoch, **assignment, sequence=1, operation='turn.acquire',
                     payload={'holder': 'lost-turn', 'ttl_seconds': 300})
        assert rt.mutate_worker_execution(db, **lease)['value']
        assert rt.mutate_worker_execution(db, **lease)['value']
        follower = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='next', payload={})
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        args = dict(epoch=epoch, admission_id=first['admission_id'], generation=claim['generation'])
        before = db.get_session('s')
        if refuse_revoke:
            db._execute_write(lambda c: c.execute("CREATE TRIGGER refuse_revoke BEFORE UPDATE OF status ON worker_executions BEGIN SELECT RAISE(ABORT, 'revoke refused'); END"))
            with pytest.raises(sqlite3.IntegrityError, match='revoke refused'):
                rt.resolve_unknown_session_input(db, **args)
            assert rt.get_session_admission(db, admission_id=first['admission_id'])['status'] == 'unknown'
            assert not db.try_acquire_session_turn_lease('s', 'follower')
            assert db.get_session('s') == before
            db._execute_write(lambda c: c.execute('DROP TRIGGER refuse_revoke'))
        result = rt.resolve_unknown_session_input(db, **args)
        assert result['status'] == 'terminal' and result['outcome'] == 'interrupted'
        assert rt.resolve_unknown_session_input(db, **args) == result
        assert db.try_acquire_session_turn_lease('s', 'follower')
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.adopt_worker_execution(db, epoch=epoch, **assignment, adoption_secret='private')
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch'):
            append(db, epoch=epoch, **assignment, sequence=2, content='late')
        assert db.get_messages('s') == []
        assert rt.claim_session_input(db, epoch=epoch, session_id='s')['admission_id'] == follower['admission_id']
    finally:
        db.close()


@pytest.mark.parametrize('replacement', ['unbound', 'other-execution'])
def test_resolution_preserves_unrelated_live_leases(tmp_path, replacement):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        db.create_session('other', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        first = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                                       request_id='first', payload={})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assignment = dict(execution_id='worker', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='secret')
        assert rt.mutate_worker_execution(db, epoch=epoch, **assignment, sequence=1,
            operation='turn.acquire', payload={'holder': 'shared', 'ttl_seconds': 300})['value']
        if replacement == 'unbound':
            # A successful local claim of the same holder drops the worker association.
            assert db.try_acquire_session_turn_lease('s', 'shared')
            target = 's'
        else:
            prior = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='other',
                                           request_id='prior', payload={})
            prior_claim = rt.claim_session_input(db, epoch=epoch, session_id='other')
            rt.settle_session_input(db, epoch=epoch, admission_id=prior['admission_id'],
                                    generation=prior_claim['generation'], outcome='completed')
            rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='other',
                                   request_id='next', payload={})
            other_claim = rt.claim_session_input(db, epoch=epoch, session_id='other')
            other = dict(execution_id='other-worker', session_id='other', generation=other_claim['generation'])
            rt.register_worker_execution(db, epoch=epoch, **other, kind='compute', adoption_secret='secret')
            assert rt.mutate_worker_execution(db, epoch=epoch, **other, sequence=1,
                operation='turn.acquire', payload={'holder': 'shared', 'ttl_seconds': 300})['value']
            target = 'other'
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        rt.resolve_unknown_session_input(db, epoch=epoch, admission_id=first['admission_id'],
                                         generation=claim['generation'])
        assert not db.try_acquire_session_turn_lease(target, 'contender')
        db.release_session_turn_lease(target, 'shared')
        assert db.try_acquire_session_turn_lease(target, 'contender')
    finally:
        db.close()


def test_resolution_preserves_lease_from_other_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        old = dict(execution_id='old', session_id='s', generation=0)
        rt.register_worker_execution(db, epoch=epoch, **old, kind='compute', adoption_secret='secret')
        assert rt.mutate_worker_execution(db, epoch=epoch, **old, sequence=1,
            operation='turn.acquire', payload={'holder': 'old-turn', 'ttl_seconds': 300})['value']
        rt.mutate_worker_execution(db, epoch=epoch, **old, sequence=2,
                                   operation='execution.finish', payload={})
        first = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                                       request_id='first', payload={})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        current = dict(execution_id='current', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, epoch=epoch, **current, kind='compute', adoption_secret='secret')
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        rt.resolve_unknown_session_input(db, epoch=epoch, admission_id=first['admission_id'],
                                         generation=claim['generation'])
        assert not db.try_acquire_session_turn_lease('s', 'contender')
        db.release_session_turn_lease('s', 'old-turn')
        assert db.try_acquire_session_turn_lease('s', 'contender')
    finally:
        db.close()


def test_unknown_resolution_retires_rotated_worker_lease(tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore

    db = SessionDB(db_path=tmp_path / 'state.db')
    store = None
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        first = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                                       request_id='first', payload={})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        scope = dict(epoch=epoch, execution_id='worker', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
        store = RuntimeSessionStore(lambda method, **p: rt.mutate_worker_execution(db, **p),
                                    scope, tmp_path / 'outbox')
        assert store.try_acquire_session_turn_lease('s', 'lost-turn')
        assert store.try_acquire_compression_lock('s', 'compressor')
        store.publish_compression_child(parent_session_id='s', child_session_id='child', source='test',
            messages=[{'role': 'assistant', 'content': 'summary'}], compression_lock_holder='compressor')
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        rt.resolve_unknown_session_input(db, epoch=epoch, admission_id=first['admission_id'],
                                         generation=claim['generation'])
        assert db.try_acquire_session_turn_lease('child', 'follower')
    finally:
        if store is not None:
            store.failure = None
            store.journal['pending'] = []
            store.close()
        db.close()


def test_raw_admission_lease_fences_stale_claim_and_same_holder(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        accepted = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                                          request_id='first', payload={})
        row = rt.claim_session_input(db, epoch=epoch, session_id='s')
        scope = (epoch, accepted['admission_id'], row['generation'])
        assert db.try_acquire_session_turn_lease('s', 'same')
        assert not db.try_acquire_session_turn_lease('s', 'same', _admission_scope=scope)
        db.release_session_turn_lease('s', 'same')
        assert db.try_acquire_session_turn_lease('s', 'same', _admission_scope=scope)
        assert db.refresh_session_turn_lease('s', 'same', _admission_scope=scope)
        assert db.try_acquire_session_turn_lease('s', 'same')
        assert not db.refresh_session_turn_lease('s', 'same', _admission_scope=scope)
        db.release_session_turn_lease('s', 'same', _admission_scope=scope)
        assert not db.try_acquire_session_turn_lease('s', 'other')
        db.release_session_turn_lease('s', 'same')
        assert db.try_acquire_session_turn_lease('s', 'same', _admission_scope=scope)
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        assert not db.refresh_session_turn_lease('s', 'same', _admission_scope=scope)
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch'):
            db.try_acquire_session_turn_lease('s', 'same', _admission_scope=scope)
        rt.resolve_unknown_session_input(db, epoch=epoch, admission_id=accepted['admission_id'],
                                         generation=row['generation'])
        assert db.try_acquire_session_turn_lease('s', 'follower')
    finally:
        db.close()


def test_raw_admission_lease_survives_long_compression_lineage(tmp_path):
    import threading
    from types import SimpleNamespace
    from agent.turn_facade_lease import admission_turn_scope, admit_durable_turn_lease

    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('segment-0', source='test')
        for index in range(1, 106):
            db.end_session(f'segment-{index - 1}', 'compression')
            db.create_session(f'segment-{index}', source='test', parent_session_id=f'segment-{index - 1}')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        first = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='segment-0',
                                       request_id='first', payload={})
        row = rt.claim_session_input(db, epoch=epoch, session_id='segment-0')
        agent = SimpleNamespace(_session_db=db, session_id='segment-105',
                                _liveness_activity_lock=lambda: threading.Lock())
        token = admission_turn_scope.set((db, epoch, row['admission_id'], row['generation'], 'segment-0'))
        try:
            turn = admit_durable_turn_lease(agent, session_id='segment-105', relay_turn_id='lost',
                task_context={'session_id': 'segment-105', 'platform': 'api_server'}, conversation_history=[])
        finally:
            admission_turn_scope.reset(token)
        assert turn.lease is not None
        assert not db.try_acquire_session_turn_lease('segment-0', 'follower')
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        rt.resolve_unknown_session_input(db, epoch=epoch, admission_id=first['admission_id'],
                                         generation=row['generation'])
        assert db.try_acquire_session_turn_lease('segment-105', 'follower')
        turn.lease.release()
    finally:
        db.close()
