// Whether the scanner looks for pages (scanic) or only for text. "auto": a page gets its outline, otherwise the
// text near the centre gets a box. "text": never look for a page; a screen, sign or poster is a bright
// rectangle that the page detector can mistake for a document. Remembered on this device only.
export type ScanMode = 'auto' | 'text';

const KEY = 'ikan.scanMode';

export function getScanMode(): ScanMode {
  try {
    return localStorage.getItem(KEY) === 'text' ? 'text' : 'auto';
  } catch {
    return 'auto'; // private mode / blocked storage: the default
  }
}

export function setScanMode(mode: ScanMode): void {
  try {
    localStorage.setItem(KEY, mode);
  } catch {
    // not remembered; the choice still applies to this scan
  }
}
