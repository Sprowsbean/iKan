export interface Reading {
  kana: string;
  romaji: string;
  /** this is the reading the word uses (when it could be worked out) */
  used: boolean;
}

/** one kanji of a word, from KANJIDIC2 */
export interface CharInfo {
  char: string;
  meaning: string;
  on: Reading[];
  kun: Reading[];
  level: string;
  strokes: number | null;
}

export interface KanjiCard {
  kanji: string;
  romaji: string;
  readings: string;
  meaning: string;
  context: string;
  level: string;
  /** missing from an older backend */
  hiragana?: string;
  chars?: CharInfo[];
  /** the word in modern forms when it was printed in old ones (戰爭 -> 戦争) */
  modern?: string;
  /** kanji the reader was unsure of, with the other characters it considered */
  unsure?: UnsureChar[];
}

export interface UnsureChar {
  char: string;
  conf: number;
  alternatives: string[];
}
