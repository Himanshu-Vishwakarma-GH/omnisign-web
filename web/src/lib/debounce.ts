import { CONFIDENCE_THRESHOLD, DEBOUNCE_FRAMES } from './constants'

/**
 * Debounce filter: a word is committed only when the same top class is
 * predicted with confidence > 0.85 for MORE than 5 consecutive passes
 * (i.e. on the 6th consecutive pass). After a commit the filter re-arms
 * only once the streak breaks, so one continuous sign is spoken once.
 */
export class Debouncer {
  private lastLabel: string | null = null
  private streak = 0
  private armed = true

  /** Returns the committed label, or null if not yet committed. */
  push(label: string, confidence: number): string | null {
    if (confidence <= CONFIDENCE_THRESHOLD || label !== this.lastLabel) {
      this.lastLabel = confidence > CONFIDENCE_THRESHOLD ? label : null
      this.streak = confidence > CONFIDENCE_THRESHOLD ? 1 : 0
      this.armed = true
      return null
    }
    this.streak += 1
    if (this.armed && this.streak > DEBOUNCE_FRAMES) {
      this.armed = false
      return label
    }
    return null
  }

  reset(): void {
    this.lastLabel = null
    this.streak = 0
    this.armed = true
  }
}
