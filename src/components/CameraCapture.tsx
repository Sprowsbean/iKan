import { useEffect, useRef, useState } from 'react';
import { X, Circle } from 'lucide-react';
import { findDocument, findTextBox } from '../services/scanner';
import { ScannerLoading } from './ScannerLoading';
import type { Box, CornerPoints } from '../services/scanner';

/** what the live detection saw when the photo was taken (both in the photo's own coordinates) */
export interface Found {
  /** a page/document's corners in pixels (scanic ML detector), or null */
  doc: CornerPoints | null;
  /** the text near the centre, as fractions of the photo (OCR text detector), or null */
  text: Box | null;
}

interface CameraCaptureProps {
  onCapture: (file: File, found: Found) => void;
  onClose: () => void;
}

const DOC_EVERY_MS = 250;   // scanic's ML detector: ~20 ms per frame on a PC
const TEXT_EVERY_MS = 900;  // the text detector is heavier (~0.4 s on a PC), so less often
const LOST_AFTER = 3;       // misses in a row before a shown border disappears (stops flicker)
const GUIDE: Box = { x: 0.08, y: 0.25, w: 0.84, h: 0.5 }; // shown when nothing is found yet
// OCR recognizers read text lines at about 48 px high; characters under half that in the camera frame read
// poorly (backend/eval: the lowest-resolution quarter of real text lines read 26% right, the best 80%)
const SMALL_TEXT_PX = 24;

// ImageCapture (full-resolution still photos) is in Chrome/Android but not in TypeScript's DOM types
type PhotoCapture = { takePhoto(): Promise<Blob> };
type PhotoCaptureCtor = new (track: MediaStreamTrack) => PhotoCapture;

/** A full-resolution still from the camera, when the browser offers it and it shows the same view as the
 *  video (same shape), so the page outline / text box found live still fit. Otherwise null. */
async function fullResPhoto(stream: MediaStream | null, vw: number, vh: number) {
  const Ctor = (window as unknown as { ImageCapture?: PhotoCaptureCtor }).ImageCapture;
  const track = stream?.getVideoTracks()[0];
  if (!Ctor || !track) return null;
  try {
    const blob = await new Ctor(track).takePhoto();
    const bmp = await createImageBitmap(blob); // applies the photo's EXIF rotation, like CropView's <img>
    const { width: w, height: h } = bmp;
    bmp.close();
    if (w <= vw || Math.abs(w / h - vw / vh) > 0.02 * (vw / vh)) return null;
    return { blob, scale: w / vw };
  } catch {
    return null; // iOS Safari and others: no ImageCapture or takePhoto fails; use the video frame
  }
}

export function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // a ref, not state: the unmount cleanup must see the stream that was opened after mount
  const streamRef = useRef<MediaStream | null>(null);
  const foundRef = useRef<Found>({ doc: null, text: null });
  const [found, setFound] = useState<Found>({ doc: null, text: null });
  const [error, setError] = useState<string>('');
  // where the video frame actually sits on screen (object-cover crops it), to draw the border on top
  const [view, setView] = useState({ ox: 0, oy: 0, w: 1, h: 1, vw: 1, vh: 1 });

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  useEffect(() => {
    let cancelled = false;
    const startCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('Camera needs HTTPS (or localhost). Use "Upload File" instead.');
        return;
      }
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          // ask for a sharp frame; the default 640x480 is too small for OCR
          video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
        });
        if (cancelled) {
          // unmounted while the permission prompt was open
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = mediaStream;
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        setError('Unable to access camera. Please check permissions.');
        console.error('Camera error:', err);
      }
    };
    startCamera();
    return () => {
      cancelled = true;
      stopCamera();
    };
  }, []);

  // live detection, like the iPhone scanner: a page gets its outline; otherwise the text near the centre
  // gets a box. Each detector runs only when its previous run has finished, so a slow phone never piles up.
  useEffect(() => {
    let stopped = false, docMiss = 0, textMiss = 0, docBusy = false, textBusy = false;
    const update = (next: Partial<Found>) => {
      foundRef.current = { ...foundRef.current, ...next };
      setFound(foundRef.current);
    };
    const ready = () => {
      const v = videoRef.current;
      if (!v || !v.videoWidth) return null;
      const r = v.getBoundingClientRect();
      const scale = Math.max(r.width / v.videoWidth, r.height / v.videoHeight);
      const w = v.videoWidth * scale, h = v.videoHeight * scale;
      setView({ ox: (r.width - w) / 2, oy: (r.height - h) / 2, w, h, vw: v.videoWidth, vh: v.videoHeight });
      return v;
    };
    const docTimer = setInterval(async () => {
      const v = ready();
      if (!v || docBusy) return;
      docBusy = true;
      const doc = await findDocument(v);
      docBusy = false;
      if (stopped) return;
      if (doc) { docMiss = 0; update({ doc }); } else if (++docMiss >= LOST_AFTER) update({ doc: null });
    }, DOC_EVERY_MS);
    const textTimer = setInterval(async () => {
      const v = ready();
      if (!v || textBusy || foundRef.current.doc) return; // a page outline wins; no need for the text box
      textBusy = true;
      const text = await findTextBox(v, v.videoWidth, v.videoHeight);
      textBusy = false;
      if (stopped) return;
      if (text) { textMiss = 0; update({ text }); } else if (++textMiss >= LOST_AFTER) update({ text: null });
    }, TEXT_EVERY_MS);
    return () => { stopped = true; clearInterval(docTimer); clearInterval(textTimer); };
  }, []);

  const busyRef = useRef(false);
  const capturePhoto = async () => {
    if (!videoRef.current || !canvasRef.current || busyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;
    busyRef.current = true;
    const seen = foundRef.current;

    // Prefer the camera's full-resolution photo (often 2-4x the video's pixels: small text stays readable)
    const full = await fullResPhoto(streamRef.current, video.videoWidth, video.videoHeight);
    if (full) {
      const s = full.scale, d0 = seen.doc;
      const at = (p: { x: number; y: number }) => ({ x: p.x * s, y: p.y * s });
      const doc = d0 && { topLeft: at(d0.topLeft), topRight: at(d0.topRight),
                          bottomRight: at(d0.bottomRight), bottomLeft: at(d0.bottomLeft) };
      stopCamera();
      onCapture(new File([full.blob], 'camera-capture.jpg', { type: full.blob.type || 'image/jpeg' }),
        { doc, text: seen.text });
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      busyRef.current = false;
      if (blob) {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        stopCamera();
        onCapture(file, seen);
      }
    }, 'image/jpeg', 0.95);
  };

  const handleClose = () => {
    stopCamera();
    onClose();
  };

  // overlay in screen pixels
  const px = (x: number, y: number) => `${view.ox + (x / view.vw) * view.w},${view.oy + (y / view.vh) * view.h}`;
  const fr = (b: Box) => [[b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h]]
    .map(([x, y]) => `${view.ox + x * view.w},${view.oy + y * view.h}`).join(' ');
  const d = found.doc;
  const pts = d ? [d.topLeft, d.topRight, d.bottomRight, d.bottomLeft].map((p) => px(p.x, p.y)).join(' ')
                : fr(found.text ?? GUIDE);
  const small = !d && found.text?.charPx !== undefined && found.text.charPx < SMALL_TEXT_PX;
  const hint = d ? 'Page found: hold still and tap the button'
             : small ? 'The text is small: move closer for a better read'
             : found.text ? 'Text found: tap the button, you can adjust the box next'
             : 'Center the text you want to scan';

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="flex-1 relative overflow-hidden">
        {error ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-white text-center px-4">{error}</p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <polygon
                points={pts}
                fill={d || found.text ? 'rgba(250,204,21,0.18)' : 'none'}
                stroke={d || found.text ? '#facc15' : 'rgba(255,255,255,0.8)'}
                strokeWidth={3}
                strokeDasharray={d || found.text ? undefined : '10 8'}
                strokeLinejoin="round"
              />
            </svg>
            <div className="absolute top-4 inset-x-0 flex flex-col items-center gap-2 px-4">
              <p className="text-center text-white text-sm drop-shadow">{hint}</p>
              <ScannerLoading />
            </div>
            <canvas ref={canvasRef} className="hidden" />
          </>
        )}
      </div>

      <div className="bg-black bg-opacity-90 p-6 flex items-center justify-between">
        <button
          onClick={handleClose}
          className="p-3 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
          aria-label="Close camera"
        >
          <X className="w-6 h-6 text-white" />
        </button>

        <button
          onClick={capturePhoto}
          disabled={!!error}
          className="p-4 bg-[#8DB88D] hover:bg-[#7DA87D] rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Capture photo"
        >
          <Circle className="w-12 h-12 text-matcha-text" strokeWidth={2} />
        </button>

        <div className="w-14"></div>
      </div>
    </div>
  );
}
