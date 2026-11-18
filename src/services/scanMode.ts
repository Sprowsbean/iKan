// How the scanner frames the text.
// - "auto": like the iPhone scanner. scanic finds a page and straightens it; with no page, a box goes around the
//   text near the centre.
// - "photo": no page finder and no box. The whole picture is sent and the OCR finds the text anywhere in it
//   (a screen, a sign, a poster: a bright rectangle the page finder could take for a document).
// Remembered on this device only.
export type ScanMode = 'auto' | 'photo';

const KEY = 'ikan.scanMode';

export function getScanMode(): ScanMode {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'photo' || v === 'text' ? 'photo' : 'auto'; // "text" was this mode's first name
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
