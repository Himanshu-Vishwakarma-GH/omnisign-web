/** Shared constants for OmniSign. */
export const SIGNS = [
  'Help', 'Doctor', 'Pain', 'Water', 'Yes', 'No', 'Family', 'Emergency',
  'Hospital', 'Medicine', 'Stop', 'Wait', 'Call', 'Police', 'Fire', 'Hurt',
  'Breathe', 'Allergy', 'ThankYou', 'Please',
] as const

export type SignWord = (typeof SIGNS)[number]

export const FRAMES = 30
export const LANDMARKS = 48
export const COORDS = 3
export const FEATURES_PER_FRAME = LANDMARKS * COORDS // 144
export const CONFIDENCE_THRESHOLD = 0.85
export const DEBOUNCE_FRAMES = 5 // commit only after MORE than 5 sequential passes
