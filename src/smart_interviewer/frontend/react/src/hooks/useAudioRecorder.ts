import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AudioSelection } from "../types";

type RecorderStatus = "idle" | "recording" | "processing";

const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/mpeg",
];

function chooseMimeType(): string | undefined {
  if (typeof window === "undefined" || typeof window.MediaRecorder === "undefined") {
    return undefined;
  }

  return MIME_CANDIDATES.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

function extensionFromMimeType(mimeType: string): string {
  if (mimeType.includes("mp4")) {
    return "m4a";
  }
  if (mimeType.includes("mpeg")) {
    return "mp3";
  }
  if (mimeType.includes("wav")) {
    return "wav";
  }
  return "webm";
}

export function useAudioRecorder() {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopResolverRef = useRef<{
    resolve: (value: AudioSelection | null) => void;
    reject: (reason?: unknown) => void;
  } | null>(null);

  const supported = useMemo(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return typeof window.MediaRecorder !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
  }, []);

  const cleanupStream = useCallback(() => {
    for (const track of streamRef.current?.getTracks() || []) {
      track.stop();
    }
    streamRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    if (!supported) {
      setError("This browser does not expose MediaRecorder. Upload an audio file instead.");
      return false;
    }

    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = chooseMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setError("The recorder failed to capture audio.");
      };

      recorder.onstop = () => {
        const finalMimeType =
          recorder.mimeType || chunksRef.current[0]?.type || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: finalMimeType });
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const extension = extensionFromMimeType(finalMimeType);
        const selection: AudioSelection = {
          blob,
          fileName: `answer-${timestamp}.${extension}`,
          mimeType: finalMimeType,
          size: blob.size,
          source: "recording",
          previewUrl: URL.createObjectURL(blob),
        };

        stopResolverRef.current?.resolve(selection);
        stopResolverRef.current = null;
        chunksRef.current = [];
        recorderRef.current = null;
        cleanupStream();
        setStatus("idle");
      };

      recorder.start();
      setStatus("recording");
      return true;
    } catch (caughtError) {
      setStatus("idle");
      cleanupStream();
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Microphone access failed.",
      );
      return false;
    }
  }, [cleanupStream, supported]);

  const stopRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return null;
    }

    setStatus("processing");

    return new Promise<AudioSelection | null>((resolve, reject) => {
      stopResolverRef.current = { resolve, reject };
      recorder.stop();
    });
  }, []);

  useEffect(() => {
    return () => {
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      cleanupStream();
    };
  }, [cleanupStream]);

  return {
    supported,
    status,
    error,
    startRecording,
    stopRecording,
  };
}
