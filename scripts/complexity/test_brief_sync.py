"""The results-sync status becomes page checks without leaking private detail."""
import datetime
import unittest

from brief_data import coherence, sync_checks

NOW = datetime.datetime(2026, 10, 20, 15, 0, tzinfo=datetime.timezone.utc)


def blocked(**kw):
    return dict(dict(blocked=True, updated_utc='2026-10-20T14:55:00+00:00', stalled_since='2026-10-20T14:50:00+00:00',
                     reason='push refused by dashboard guard', blocked_cycles=2), **kw)


class SyncChecks(unittest.TestCase):
    def test_healthy_or_absent_status_adds_nothing(self):
        self.assertEqual(sync_checks(None, NOW), [])
        self.assertEqual(sync_checks(dict(state='committed and remote verified', updated_utc='2026-10-20T14:55:00+00:00'), NOW), [])

    def test_blocked_escalates_to_error_after_fifteen_minutes(self):
        [early] = sync_checks(blocked(), NOW)
        self.assertEqual((early['level'], early['check']), ('warn', 'results sync blocked'))
        self.assertIn('push refused by dashboard guard', early['detail'])
        [late] = sync_checks(blocked(stalled_since='2026-10-20T13:05:00+00:00'), NOW)
        self.assertEqual(late['level'], 'error')

    def test_stopped_timer_and_unreadable_status_are_flagged(self):
        [stopped] = sync_checks(dict(state='no new measured evidence', updated_utc='2026-10-20T14:00:00+00:00'), NOW)
        self.assertEqual((stopped['level'], stopped['check']), ('error', 'results sync not running'))
        [unknown] = sync_checks(dict(state='unreachable'), NOW)
        self.assertEqual(unknown['level'], 'warn')

    def test_private_detail_never_reaches_the_page(self):
        secret = 'host 203.0.113.7 matched'
        for status in (blocked(detail=secret, first_detail=secret, error=secret), blocked(reason=secret)):
            [check] = sync_checks(status, NOW)
            self.assertNotIn('203.0.113.7', check['detail'])
        [check] = sync_checks(blocked(reason=secret), NOW)
        self.assertIn('unclassified failure', check['detail'])


    def test_quiet_sync_warns_only_while_a_live_host_has_pending_cells(self):
        def stale(coverage):
            perf = dict(synced_utc='2026-10-01T00:00:00+00:00', coverage=coverage)
            repro = dict(checks=[], july_columns=[], misses=[], sets=[])
            return [c for c in coherence([], perf, repro, {}) if c['check'] == 'no new timings']
        done = dict(ok=9, cells=10, refused=1, failed=0, pending=0)
        self.assertEqual(stale(dict(a=done, b=dict(done, pending=4, retired=dict(since='2026-10-20')))), [])
        self.assertEqual(len(stale(dict(a=dict(done, pending=1)))), 1)
        self.assertEqual(len(stale({})), 1)


if __name__ == '__main__':
    unittest.main()
