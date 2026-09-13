/**
 * "Load a sample profile" -- and the honest label of what will be scored.
 *
 * Loading a row sets `sampleUserId`, which makes the next submit post
 * `{ user_id }` and score the stored row. Editing any field clears it, and the
 * submit posts the edited fields instead. Both are correct; which one is about
 * to happen is not something the user should have to infer, so this component
 * states it.
 */

import { useState } from 'react'

import { useProfile } from '../../context/ProfileContext'
import { ApiError, getUser } from '../../lib/api'
import { shortUserId } from '../../lib/format'
import { Badge, Button, cx } from '../ui/primitives'

/** users.csv holds U00001..U05000; pick one at random for the demo button. */
function randomUserId(min: string, max: string): number {
  const low = Number(min.replace(/\D/g, '')) || 1
  const high = Number(max.replace(/\D/g, '')) || 5000
  return low + Math.floor(Math.random() * (high - low + 1))
}

export function SampleLoader({
  idRange,
}: {
  idRange: { min: string; max: string }
}) {
  const { loadSample, sampleUserId, loadedFrom, source, reset } = useProfile()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      loadSample(await getUser(randomUserId(idRange.min, idRange.max)))
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : 'Could not load a sample profile.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="border border-ink-600 bg-ink-800 p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="eyebrow text-ash-dim">Demo shortcut</p>
          <p className="mt-1 text-sm text-ash-muted">
            Fill all three steps from a real profile in users.csv.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={load}
            disabled={loading}
            type="button"
          >
            {loading ? 'Loading…' : loadedFrom ? 'Load another' : 'Load a sample profile'}
          </Button>
          {loadedFrom && (
            <Button variant="ghost" size="sm" onClick={reset} type="button">
              Clear
            </Button>
          )}
        </div>
      </div>

      {error && (
        <p role="alert" className="mt-3 text-sm text-crimson-500">
          {error}
        </p>
      )}

      {loadedFrom && (
        <div
          className={cx(
            'mt-4 border-l-2 pl-3',
            sampleUserId ? 'border-crimson-600' : 'border-ink-600',
          )}
        >
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={sampleUserId ? 'crimson' : 'neutral'}>
              Profile #{shortUserId(loadedFrom.user_id)}
            </Badge>
            <Badge tone="outline">
              {loadedFrom.qualifying_ratings} rating
              {loadedFrom.qualifying_ratings === 1 ? '' : 's'} of 4+
            </Badge>
          </div>

          {/*
            The distinction that matters. `user_id` in the request body means
            "score the stored row and ignore everything else", so an edited
            form must not carry it -- and the user is told which request is
            queued up rather than left to guess.
          */}
          <p className="mt-2 text-xs leading-relaxed text-ash-muted">
            {source === 'sample' ? (
              <>
                Submitting now scores the <strong className="text-ash-dim">stored</strong>{' '}
                profile {loadedFrom.user_id} by id, so its rating history counts
                towards the scoring mode. Edit any field and your edits are
                scored instead.
              </>
            ) : (
              <>
                You have edited this profile, so submitting scores{' '}
                <strong className="text-ash-dim">your version</strong> — the
                stored row {loadedFrom.user_id} and its rating history are no
                longer part of the request.
              </>
            )}
          </p>
        </div>
      )}
    </div>
  )
}
