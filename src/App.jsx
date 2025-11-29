import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Mic,
  Square,
  Activity,
  Send,
  AlertCircle,
  Volume2,
  Play,
  Plus,
  Pencil,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// --- HARD-CODED REALTIME API CONFIG (FROM ENV VARS) ---
const HARD_CODED_BASE_URL = import.meta.env.VITE_BASE_URL;
const HARD_CODED_API_KEY = import.meta.env.VITE_API_KEY;

// --- Gemini API Constants ---
const GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025';
const TTS_API_URL_BASE = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent`;

// --- System Prompt Presets ---
const DEFAULT_SYSTEM_PROMPTS = [
  {
    id: 'meeting_assistant',
    name: 'Meeting Assistant',
    prompt:
      'You are a helpful meeting assistant. Transcribe the meeting and answer questions concisely in markdown. Use markdown tables, bulleted lists, or headers for structured data.',
  },
  {
    id: 'qa_assistant',
    name: 'Q&A Assistant',
    prompt:
      'You are a concise question-answering assistant. Provide direct, well-structured answers in markdown and ask clarifying questions if needed.',
  },
  {
    id: 'brainstorm_partner',
    name: 'Brainstorm Partner',
    prompt:
      'You are a creative brainstorming partner. Generate multiple ideas, organize them into bullet lists, and highlight the most promising options.',
  },
];

// --- Utility: Audio Resampler & Converter for Realtime API ---
const floatTo16BitPCM = (float32Array) => {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const dataView = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    dataView.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
};

const base64EncodeAudio = (float32Array) => {
  const pcmBuffer = floatTo16BitPCM(float32Array);
  let binary = '';
  const bytes = new Uint8Array(pcmBuffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
};

// --- Utility: TTS Audio Conversion (PCM to WAV) ---
const base64ToArrayBuffer = (base64) => {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};

const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

const pcmToWav = (pcmData, sampleRate) => {
  const buffer = new ArrayBuffer(44 + pcmData.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + pcmData.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, pcmData.length * 2, true);

  let offset = 44;
  for (let i = 0; i < pcmData.length; i++, offset += 2) {
    view.setInt16(offset, pcmData[i], true);
  }

  return new Blob([view], { type: 'audio/wav' });
};

// --- Generic API Caller with Backoff (for Gemini) ---
const callGeminiApi = async (url, payload, maxRetries = 5) => {
  const apiKey = ''; // Leave empty if Canvas/runtime injects it; otherwise add your Gemini key
  let lastError = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const fetchUrl = `${url}?key=${apiKey}`;
      const response = await fetch(fetchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        return response;
      } else if (response.status === 429 || response.status >= 500) {
        lastError = new Error(`API request failed with status ${response.status}.`);
        const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      } else {
        const errorBody = await response.json();
        throw new Error(
          `Gemini API error: ${response.status} - ${errorBody.error?.message || 'Unknown Error'}`
        );
      }
    } catch (error) {
      lastError = error;
      const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error(
    `Gemini API call failed after ${maxRetries} attempts: ${lastError.message || lastError}`
  );
};

// --- Config Utility: Load and Save from Local Storage ---
const getInitialConfig = () => {
  const fallback = {
    systemPrompts: DEFAULT_SYSTEM_PROMPTS,
    selectedPromptId: DEFAULT_SYSTEM_PROMPTS[0].id,
  };

  try {
    const savedConfig = localStorage.getItem('voiceAppConfig');
    if (!savedConfig) return fallback;

    const parsed = JSON.parse(savedConfig);

    let systemPrompts = Array.isArray(parsed.systemPrompts)
      ? [...parsed.systemPrompts]
      : [...DEFAULT_SYSTEM_PROMPTS];

    const existingIds = new Set(systemPrompts.map((p) => p.id));
    DEFAULT_SYSTEM_PROMPTS.forEach((preset) => {
      if (!existingIds.has(preset.id)) {
        systemPrompts.push(preset);
      }
    });

    const selectedPromptId =
      systemPrompts.find((p) => p.id === parsed.selectedPromptId)?.id ||
      systemPrompts[0]?.id ||
      DEFAULT_SYSTEM_PROMPTS[0].id;

    return { systemPrompts, selectedPromptId };
  } catch (e) {
    console.error('Failed to parse stored config', e);
    return fallback;
  }
};

// --- FIXED GFM TABLE PARSER (Line-by-Line Iteration) ---
const convertGfmTablesToHtml = (markdownText) => {
  const lines = markdownText.split(/\r?\n/);
  const parts = [];
  let i = 0;

  const isSeparator = (line) => /^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$/.test(line);

  const extractCells = (row) =>
    row
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((c) => c.trim());

  const getAlignmentClass = (cell) => {
    const left = cell.startsWith(':');
    const right = cell.endsWith(':');
    if (left && right) return 'text-center';
    if (right) return 'text-right';
    return 'text-left';
  };

  while (i < lines.length) {
    const line = lines[i];

    if (!line.includes('|') || !lines[i + 1] || !isSeparator(lines[i + 1])) {
      parts.push({ type: 'text', content: line + '\n' });
      i++;
      continue;
    }

    const header = extractCells(lines[i]);
    const separator = extractCells(lines[i + 1]);

    const alignments = separator.map(getAlignmentClass);
    const colCount = header.length;

    i += 2;

    const body = [];
    while (i < lines.length && lines[i].includes('|')) {
      let rawCells = extractCells(lines[i].replace(/\|\|/g, '|'));

      const rowCells = rawCells.slice(0, colCount);
      while (rowCells.length < colCount) {
        rowCells.push('');
      }

      body.push(rowCells);
      i++;
    }

    let html =
      '<div class="overflow-x-auto"><table class="min-w-full divide-y divide-slate-700 mt-4 border border-slate-700 rounded-lg">';

    html += '<thead class="bg-slate-800"><tr>';
    header.slice(0, colCount).forEach((h, idx) => {
      const alignClass = alignments[idx] || 'text-left';
      html += `<th class="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-white border-b border-slate-700 ${alignClass}">${h}</th>`;
    });
    html += '</tr></thead>';

    html += '<tbody>';
    body.forEach((row) => {
      html += '<tr class="hover:bg-slate-800/50 transition-colors">';
      row.forEach((cell, idx) => {
        const alignClass = alignments[idx] || 'text-left';
        html += `<td class="px-4 py-2 whitespace-normal text-sm text-slate-300 border-t border-slate-700 ${alignClass}">${cell}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table></div>';

    parts.push({ type: 'html', content: html });
  }

  const mergedParts = [];
  let currentText = '';

  parts.forEach((part) => {
    if (part.type === 'text') {
      currentText += part.content;
    } else {
      if (currentText) {
        mergedParts.push({ type: 'text', content: currentText.trimStart() });
        currentText = '';
      }
      mergedParts.push(part);
    }
  });

  if (currentText) {
    mergedParts.push({ type: 'text', content: currentText.trimStart() });
  }

  return mergedParts;
};

export default function App() {
  // --- State ---
  const [config, setConfig] = useState(getInitialConfig);
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isMicReady, setIsMicReady] = useState(false);
  const [error, setError] = useState(null);
  const [volume, setVolume] = useState(0);
  const [serverStatus, setServerStatus] = useState('Idle');
  const [inputText, setInputText] = useState('');
  const [ttsLoadingId, setTtsLoadingId] = useState(null);
  const [items, setItems] = useState([]);

  // Prompt panel (add / edit) state
  const [isPromptPanelOpen, setIsPromptPanelOpen] = useState(false);
  const [promptPanelMode, setPromptPanelMode] = useState('add'); // 'add' | 'edit'
  const [newPromptName, setNewPromptName] = useState('');
  const [newPromptText, setNewPromptText] = useState('');

  // --- Refs ---
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const messagesEndRef = useRef(null);
  const startTimeRef = useRef(null);
  const audioPlayerRef = useRef(null);

  const isRecordingRef = useRef(false);

  const activePrompt =
    config.systemPrompts.find((p) => p.id === config.selectedPromptId) ||
    config.systemPrompts[0] ||
    DEFAULT_SYSTEM_PROMPTS[0];

  // Helper: current active system prompt text
  const getActiveSystemPromptText = useCallback(() => {
    return activePrompt.prompt;
  }, [activePrompt]);

  // --- Persist Config to Local Storage ---
  useEffect(() => {
    localStorage.setItem('voiceAppConfig', JSON.stringify(config));
  }, [config]);

  // --- Cleanup on Component Unmount ---
  const stopMicTracks = useCallback(() => {
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsMicReady(false);
  }, []);

  useEffect(() => {
    return () => {
      stopMicTracks();
    };
  }, [stopMicTracks]);

  // --- Helpers ---
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const formatTime = () => {
    if (!startTimeRef.current) return '00:00:00';
    const elapsedMs = Date.now() - startTimeRef.current;
    const totalSeconds = Math.floor(elapsedMs / 1000);
    const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, '0');
    const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
  };

  // --- Session instructions updater (for prompt changes while connected) ---
  const updateSessionInstructions = useCallback((newInstructions) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const updateMsg = {
        type: 'session.update',
        session: {
          instructions: newInstructions,
        },
      };
      wsRef.current.send(JSON.stringify(updateMsg));
    }
  }, []);

  // --- Audio Pipeline Functions ---
  const stopAudioProcessing = useCallback(() => {
    if (isMicReady && processorRef.current && isRecording) {
      processorRef.current.disconnect();
      setIsRecording(false);
      isRecordingRef.current = false;
      setVolume(0);
    }
  }, [isMicReady, isRecording]);

  const startAudioProcessing = useCallback(() => {
    if (isMicReady && processorRef.current && audioContextRef.current && !isRecording) {
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume().catch(console.error);
      }
      processorRef.current.connect(audioContextRef.current.destination);
      setIsRecording(true);
      isRecordingRef.current = true;
      setVolume(0);
    } else if (!isMicReady) {
      console.warn('Cannot start audio processing: Mic not ready.');
    } else if (isRecording) {
      console.warn('Cannot start audio processing: Already recording.');
    }
  }, [isMicReady, isRecording]);

  const disconnect = useCallback(() => {
    stopAudioProcessing();

    if (audioContextRef.current && audioContextRef.current.state === 'running') {
      audioContextRef.current.suspend().catch(console.error);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.src = '';
    }

    setIsConnected(false);
    setIsRecording(false);
    isRecordingRef.current = false;
    setServerStatus('Idle');
    setItems([]);
    setTtsLoadingId(null);
    startTimeRef.current = null;
  }, [stopAudioProcessing]);

  const initializeAudioPipeline = useCallback(async () => {
    if (isMicReady && audioContextRef.current) {
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      return;
    }

    try:
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      if (!audioContextRef.current) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 24000,
        });
        audioContextRef.current = audioContext;
      }

      const audioContext = audioContextRef.current;

      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      const analyser = audioContext.createAnalyser();

      analyser.fftSize = 256;
      analyserRef.current = analyser;
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!isRecordingRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN)
          return;

        const inputData = e.inputBuffer.getChannelData(0);

        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setVolume(Math.min(1, rms * 5));

        const base64Audio = base64EncodeAudio(inputData);
        wsRef.current.send(
          JSON.stringify({
            type: 'input_audio_buffer.append',
            audio: base64Audio,
          })
        );
      };

      source.connect(analyser);
      analyser.connect(processor);

      setIsMicReady(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Could not access microphone. Please check permissions.');
      setIsMicReady(false);
    }
  }, [isMicReady]);

  const toggleRecording = useCallback(() => {
    if (!isConnected || !isMicReady) {
      console.warn('Cannot toggle recording: Not connected or mic not ready.', {
        isConnected,
        isMicReady,
      });
      return;
    }

    if (isRecording) {
      stopAudioProcessing();
    } else {
      startAudioProcessing();
    }
  }, [isConnected, isMicReady, isRecording, stopAudioProcessing, startAudioProcessing]);

  // --- Connection Logic ---
  const connect = useCallback(async () => {
    if (!HARD_CODED_API_KEY || !HARD_CODED_BASE_URL) {
      setError('Server configuration is missing. Please contact your administrator.');
      return;
    }

    setError(null);

    let url = HARD_CODED_BASE_URL;

    if (url.startsWith('https://')) {
      url = url.replace('https://', 'wss://');
    } else if (url.startsWith('http://')) {
      url = url.replace('http://', 'ws://');
    }

    if (url.includes('azure') && !url.includes('api-key=')) {
      const separator = url.includes('?') ? '&' : '?';
      url = `${url}${separator}api-key=${HARD_CODED_API_KEY}`;
    }

    startTimeRef.current = Date.now();

    try {
      await initializeAudioPipeline();

      if (!audioContextRef.current) {
        setError('Failed to initialize microphone. Check permissions.');
        return;
      }

      const ws = new WebSocket(url);

      ws.onopen = async () => {
        console.log('Connected to Realtime API');
        wsRef.current = ws;
        setIsConnected(true);
        startTimeRef.current = Date.now();

        if (audioContextRef.current) {
          startAudioProcessing();
        }

        const sessionUpdate = {
          type: 'session.update',
          session: {
            instructions: getActiveSystemPromptText(),
            voice: 'alloy',
            modalities: ['text', 'audio'],
            input_audio_format: 'pcm16',
            output_audio_format: 'pcm16',
            input_audio_transcription: {
              model: 'whisper-1',
            },
            turn_detection: {
              type: 'server_vad',
            },
          },
        };
        ws.send(JSON.stringify(sessionUpdate));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerEvent(data);
      };

      ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
        setError('Connection failed. Please check server configuration.');
        disconnect();
      };

      ws.onclose = (event) => {
        console.log('Disconnected', event.code, event.reason);
        if (event.code === 1006) {
          setError(
            'Connection closed unexpectedly (1006). This often means Authentication failed or protocol mismatch.'
          );
        }
        disconnect();
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err.message);
    }
  }, [disconnect, getActiveSystemPromptText, initializeAudioPipeline, startAudioProcessing]);

  // --- Keyboard Listener (Spacebar toggles mic) ---
  useEffect(() => {
    const handleKeydown = (event) => {
      if (event.code === 'Space') {
        const activeElement = document.activeElement;
        const tagName = activeElement?.tagName;

        if (tagName === 'INPUT' || tagName === 'TEXTAREA') {
          return;
        }

        if (isConnected) {
          event.preventDefault();
          toggleRecording();
        }
      }
    };

    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [isConnected, toggleRecording]);

  // --- Text Input Logic ---
  const handleSendText = () => {
    const textToSend = inputText.trim();
    if (textToSend === '' || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const message = {
      type: 'conversation.item.create',
      item: {
        type: 'text',
        content: [{ type: 'text', value: textToSend }],
      },
    };
    wsRef.current.send(JSON.stringify(message));

    appendItem(crypto.randomUUID(), 'user', textToSend, false);
    scrollToBottom();
    setInputText('');
  };

  // --- Gemini: TTS Readback Logic ---
  const handleReadText = async (itemId, textToRead) => {
    if (ttsLoadingId === itemId) {
      if (audioPlayerRef.current) audioPlayerRef.current.pause();
      setTtsLoadingId(null);
      return;
    }

    setTtsLoadingId(itemId);
    setError(null);

    try {
      const payload = {
        contents: [{ parts: [{ text: `Say in a clear and professional tone: ${textToRead}` }] }],
        generationConfig: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Charon' } },
          },
        },
      };

      const response = await callGeminiApi(TTS_API_URL_BASE, payload);
      const result = await response.json();

      const audioPart = result.candidates?.[0]?.content?.parts?.[0];
      const audioData = audioPart?.inlineData?.data;
      const mimeType = audioPart?.inlineData?.mimeType;

      if (audioData && mimeType && mimeType.startsWith('audio/L16')) {
        const rateMatch = mimeType.match(/rate=(\d+)/);
        const sampleRate = rateMatch ? parseInt(rateMatch[1], 10) : 24000;

        const pcmData = new Int16Array(base64ToArrayBuffer(audioData));
        const wavBlob = pcmToWav(pcmData, sampleRate);
        const audioUrl = URL.createObjectURL(wavBlob);

        if (!audioPlayerRef.current) {
          audioPlayerRef.current = new Audio();
          audioPlayerRef.current.onended = () => setTtsLoadingId(null);
          audioPlayerRef.current.onerror = () => {
            console.error('Audio playback error.');
            setTtsLoadingId(null);
            setError('Failed to play audio.');
          };
        }

        if (audioPlayerRef.current.src) {
          audioPlayerRef.current.pause();
          audioPlayerRef.current.src = '';
        }

        audioPlayerRef.current.src = audioUrl;
        audioPlayerRef.current.play();
      } else {
        throw new Error('Invalid TTS audio data received.');
      }
    } catch (err) {
      console.error('Gemini TTS Error:', err);
      setError(`Text-to-Speech failed: ${err.message}`);
      setTtsLoadingId(null);
    }
  };

  // --- Event Handling (Realtime API) ---
  const handleServerEvent = (event) => {
    switch (event.type) {
      case 'input_audio_buffer.speech_started':
        setServerStatus('Listening...');
        break;

      case 'input_audio_buffer.speech_stopped':
        setServerStatus('Processing...');
        break;

      case 'conversation.item.input_audio_transcription.delta': {
        const itemId = event.item_id;
        const delta = event.delta;
        if (delta) {
          updateItem(itemId, 'user', delta);
          setServerStatus('Transcribing...');
        }
        break;
      }

      case 'conversation.item.input_audio_transcription.completed':
        if (event.transcript) {
          updateItemContent(event.item_id, 'user', event.transcript, false);
          setServerStatus('Idle');
          scrollToBottom();
        }
        break;

      case 'response.text.delta': {
        const itemId = event.item_id;
        const delta = event.delta;
        updateItem(itemId, 'assistant', delta);
        setServerStatus('Responding');
        break;
      }

      case 'response.audio_transcript.delta': {
        const itemId = event.item_id;
        const delta = event.delta;
        updateItem(itemId, 'assistant', delta);
        setServerStatus('Responding');
        break;
      }

      case 'response.done':
        setServerStatus('Idle');
        break;

      case 'error':
        console.error('Server Error:', event.error);
        setError(`API Error: ${event.error.message}`);
        setServerStatus('Error');
        break;

      default:
        break;
    }
  };

  // --- UI Data Helpers ---
  const appendItem = (id, role, text, partial) => {
    const timestamp = formatTime();
    setItems((prev) => {
      if (prev.find((i) => i.id === id && !partial)) return prev;
      return [...prev, { id, role, content: text, partial, timestamp }];
    });
  };

  const updateItem = (id, role, deltaText) => {
    setItems((prev) => {
      const existingIndex = prev.findIndex((item) => item.id === id);
      if (existingIndex !== -1) {
        const newItems = [...prev];
        newItems[existingIndex] = {
          ...newItems[existingIndex],
          content: newItems[existingIndex].content + deltaText,
          partial: true,
        };
        return newItems;
      } else {
        return [
          ...prev,
          { id, role, content: deltaText, partial: true, timestamp: formatTime() },
        ];
      }
    });
  };

  const updateItemContent = (id, role, newText, partial) => {
    setItems((prev) => {
      const existingIndex = prev.findIndex((item) => item.id === id);
      if (existingIndex !== -1) {
        const newItems = [...prev];
        newItems[existingIndex] = {
          ...newItems[existingIndex],
          content: newText,
          partial: partial,
        };
        return newItems;
      } else {
        return [
          ...prev,
          { id, role, content: newText, partial: partial, timestamp: formatTime() },
        ];
      }
    });
  };

  // --- System Prompt UI Handlers ---
  const handlePromptSelectChange = (e) => {
    const value = e.target.value;

    if (value === '__add_new__') {
      // open panel in ADD mode
      setPromptPanelMode('add');
      setNewPromptName('');
      setNewPromptText('');
      setIsPromptPanelOpen(true);
      return;
    }

    setConfig((prev) => ({
      ...prev,
      selectedPromptId: value,
    }));

    const selected =
      config.systemPrompts.find((p) => p.id === value) || config.systemPrompts[0];

    if (selected && isConnected) {
      updateSessionInstructions(selected.prompt);
    }
  };

  const handleOpenEditPromptPanel = () => {
    if (!activePrompt) return;
    setPromptPanelMode('edit');
    setNewPromptName(activePrompt.name || '');
    setNewPromptText(activePrompt.prompt || '');
    setIsPromptPanelOpen(true);
  };

  const handleCreateNewPrompt = () => {
    const name = newPromptName.trim() || 'Custom Prompt';
    const promptText =
      newPromptText.trim() ||
      'You are a helpful assistant. Answer clearly and concisely in markdown.';

    const newPrompt = {
      id: crypto.randomUUID(),
      name,
      prompt: promptText,
    };

    setConfig((prev) => {
      const updatedPrompts = [...prev.systemPrompts, newPrompt];
      return {
        ...prev,
        systemPrompts: updatedPrompts,
        selectedPromptId: newPrompt.id,
      };
    });

    if (isConnected) {
      updateSessionInstructions(promptText);
    }

    setIsPromptPanelOpen(false);
    setNewPromptName('');
    setNewPromptText('');
  };

  const handleSaveEditedPrompt = () => {
    const name = newPromptName.trim();
    const text = newPromptText.trim();

    if (!name || !text) {
      alert('Name and prompt cannot be empty.');
      return;
    }

    setConfig((prev) => {
      const updated = prev.systemPrompts.map((p) =>
        p.id === prev.selectedPromptId ? { ...p, name, prompt: text } : p
      );
      return { ...prev, systemPrompts: updated };
    });

    if (isConnected) {
      updateSessionInstructions(text);
    }

    setIsPromptPanelOpen(false);
    setNewPromptName('');
    setNewPromptText('');
  };

  const handleDeleteCurrentPrompt = () => {
    if (config.systemPrompts.length <= 1) {
      alert('At least one system prompt must remain.');
      return;
    }

    const idToDelete = config.selectedPromptId;

    const updated = config.systemPrompts.filter((p) => p.id !== idToDelete);
    const newSelected = updated[0].id;

    setConfig({
      systemPrompts: updated,
      selectedPromptId: newSelected,
    });

    const newPromptObj = updated.find((p) => p.id === newSelected);
    if (isConnected && newPromptObj) {
      updateSessionInstructions(newPromptObj.prompt);
    }

    setIsPromptPanelOpen(false);
    setNewPromptName('');
    setNewPromptText('');
  };

  // --- Rendering conversation item ---
  const renderConversationItem = (item) => {
    const speakerLabel = item.role === 'user' ? 'Interviewer' : 'Speaker';
    const isUser = item.role === 'user';

    const AssistantContent = ({ content }) => {
      const parts = convertGfmTablesToHtml(content);

      return (
        <div className="text-sm md:text-base text-slate-100 leading-relaxed pr-2 md:pr-4">
          {parts.map((part, index) => {
            if (part.type === 'html') {
              return (
                <div
                  key={index}
                  dangerouslySetInnerHTML={{ __html: part.content }}
                  className="my-3"
                />
              );
            } else {
              return (
                <ReactMarkdown
                  key={index}
                  className="prose prose-invert prose-sm md:prose-base max-w-none prose-p:leading-relaxed 
                             prose-ul:list-disc prose-ul:pl-5 prose-li:my-1
                             prose-ol:list-decimal prose-ol:pl-5
                             prose-code:bg-slate-700 prose-code:text-yellow-300 prose-code:px-1 prose-code:rounded
                             prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-700 prose-pre:p-3 prose-pre:rounded-lg"
                >
                  {part.content}
                </ReactMarkdown>
              );
            }
          })}
        </div>
      );
    };

    return (
      <div key={item.id} className="w-full flex flex-col gap-1 py-1 px-1 md:px-2">
        <div
          className={`flex items-center text-[10px] md:text-xs font-semibold ${
            isUser ? 'text-indigo-400' : 'text-slate-300'
          }`}
        >
          <span className="mr-2 uppercase tracking-wide">{speakerLabel}</span>

          {!isUser && item.content.length > 0 && (
            <button
              onClick={() => handleReadText(item.id, item.content)}
              className={`ml-2 p-1 rounded-full transition-colors ${
                ttsLoadingId === item.id
                  ? 'bg-red-600/50 text-red-300 animate-pulse'
                  : 'text-slate-500 hover:text-white hover:bg-slate-700/50'
              }`}
              title={ttsLoadingId === item.id ? 'Stop Reading' : 'Read Aloud'}
            >
              {ttsLoadingId === item.id ? (
                <Square size={12} fill="currentColor" />
              ) : (
                <Volume2 size={12} />
              )}
            </button>
          )}
        </div>

        <div className="flex items-start justify-between">
          <div className="flex-1">
            {isUser ? (
              <span
                className={`text-sm md:text-base ${
                  item.partial ? 'text-indigo-200 animate-pulse' : 'text-slate-100'
                }`}
              >
                {item.content}
              </span>
            ) : (
              <AssistantContent content={item.content} />
            )}
          </div>
          <span className="shrink-0 text-[9px] md:text-[10px] text-slate-500 mt-1 ml-3">
            {item.timestamp}
          </span>
        </div>

        <div className="border-b border-slate-800/60 mt-2" />
      </div>
    );
  };

  return (
    <>
      <style>{`
        .strict-cursor-default,
        .strict-cursor-default *,
        .strict-cursor-default button,
        .strict-cursor-default input,
        .strict-cursor-default textarea,
        .strict-cursor-default a {
            cursor: default !important;
        }
        .overflow-x-auto {
            overflow-x: auto;
        }
        .min-w-full th, .min-w-full td {
            vertical-align: top;
        }
      `}</style>

      <div className="flex flex-col h-screen bg-slate-900 text-slate-100 font-sans strict-cursor-default">
        {/* Slim Header - with status + prompt dropdown + edit + controls */}
        <header className="flex items-center justify-between px-3 md:px-4 py-1.5 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2 text-[11px] md:text-xs text-slate-400">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}
            />
            <span className="font-medium">
              {isConnected ? 'Live conversation' : 'Not connected'}
            </span>
            <span className="hidden sm:inline text-slate-500">
              {serverStatus === 'Listening...'
                ? 'Listening'
                : serverStatus === 'Transcribing...'
                ? 'Transcribing'
                : serverStatus === 'Processing...'
                ? 'Processing'
                : serverStatus === 'Responding'
                ? 'Responding'
                : 'Idle'}
            </span>
          </div>

          {/* Center: system prompt dropdown + pencil */}
          <div className="flex-1 flex justify-center px-2">
            <div className="flex items-center gap-2 max-w-md w-full">
              <span className="hidden md:inline text-[10px] uppercase text-slate-500">
                Prompt
              </span>
              <select
                value={config.selectedPromptId}
                onChange={handlePromptSelectChange}
                className="flex-1 bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-[11px] md:text-xs text-slate-200 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
              >
                {config.systemPrompts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
                <option value="__add_new__">+ Add new system prompt…</option>
              </select>
              {/* Pencil icon: edit current prompt */}
              <button
                onClick={handleOpenEditPromptPanel}
                className="p-1.5 rounded-md bg-slate-900 border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white"
                title="Edit current system prompt"
              >
                <Pencil size={12} />
              </button>
            </div>
          </div>

          {/* Right: mic + connect/disconnect */}
          <div className="flex items-center gap-2">
            {isConnected && isMicReady && (
              <button
                onClick={toggleRecording}
                className={`p-1.5 md:p-2 rounded-full transition-all text-xs ${
                  isRecording
                    ? 'bg-red-600 hover:bg-red-700 text-white shadow shadow-red-500/30'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow shadow-indigo-500/30'
                }`}
                title={isRecording ? 'Pause Mic (Spacebar)' : 'Start Mic (Spacebar)'}
              >
                {isRecording ? (
                  <Square size={14} />
                ) : (
                  <Play size={14} fill="currentColor" />
                )}
              </button>
            )}

            {!isConnected ? (
              <button
                onClick={connect}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] md:text-xs bg-indigo-600 hover:bg-indigo-700 text-white shadow shadow-indigo-500/30"
              >
                <Mic size={12} /> Connect
              </button>
            ) : (
              <button
                onClick={disconnect}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] md:text-xs bg-slate-800 hover:bg-slate-700 text-red-400 border border-red-900/60"
              >
                <Square size={12} /> Disconnect
              </button>
            )}
          </div>
        </header>

        {/* Error banner (if any) */}
        {error && (
          <div className="px-3 md:px-4 py-1.5 bg-red-950/70 border-b border-red-800 text-red-200 text-[11px] md:text-xs flex items-center gap-2">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        {/* Inline Prompt Panel under header (Add / Edit) */}
        {isPromptPanelOpen && (
          <div className="px-3 md:px-4 py-2 bg-slate-950 border-b border-slate-800">
            <div className="max-w-3xl mx-auto flex flex-col gap-2">
              <div className="flex items-center justify-between text-[11px] md:text-xs text-slate-300">
                <div className="flex items-center gap-2">
                  {promptPanelMode === 'add' ? <Plus size={12} /> : <Pencil size={12} />}
                  <span className="font-semibold">
                    {promptPanelMode === 'add' ? 'New System Prompt' : 'Edit System Prompt'}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-2">
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] uppercase text-slate-500">Name</label>
                  <input
                    type="text"
                    value={newPromptName}
                    onChange={(e) => setNewPromptName(e.target.value)}
                    placeholder="e.g., Support Bot, Coding Tutor"
                    className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-[11px] md:text-xs text-slate-200 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <label className="text-[10px] uppercase text-slate-500">
                    System Prompt
                  </label>
                  <textarea
                    value={newPromptText}
                    onChange={(e) => setNewPromptText(e.target.value)}
                    placeholder="Describe how this assistant should behave..."
                    className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-[11px] md:text-xs text-slate-200 h-16 focus:ring-2 focus:ring-indigo-500 focus:outline-none resize-none"
                  />
                </div>
              </div>

              <div className="flex justify-between items-center mt-2">
                {/* Delete in EDIT mode only */}
                {promptPanelMode === 'edit' ? (
                  <button
                    onClick={handleDeleteCurrentPrompt}
                    className="px-3 py-1.5 rounded-md text-xs bg-red-700 hover:bg-red-800 text-white border border-red-900/60"
                  >
                    Delete Prompt
                  </button>
                ) : (
                  <div />
                )}

                {/* Right side: Cancel + Save */}
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setIsPromptPanelOpen(false);
                      setNewPromptName('');
                      setNewPromptText('');
                    }}
                    className="px-3 py-1.5 rounded-md text-xs bg-slate-800 hover:bg-slate-700 text-slate-200"
                  >
                    Cancel
                  </button>

                  {promptPanelMode === 'add' ? (
                    <button
                      onClick={handleCreateNewPrompt}
                      className="px-3 py-1.5 rounded-md text-xs bg-indigo-600 hover:bg-indigo-700 text-white flex items-center gap-1"
                    >
                      <Plus size={12} /> Save & Use
                    </button>
                  ) : (
                    <button
                      onClick={handleSaveEditedPrompt}
                      className="px-3 py-1.5 rounded-md text-xs bg-indigo-600 hover:bg-indigo-700 text-white"
                    >
                      Save Changes
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Main content: full-screen conversation + footer */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Conversation area */}
          <div className="flex-1 overflow-y-auto px-3 md:px-5 py-3 md:py-4 space-y-2">
            {!isConnected && items.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-500">
                <Mic size={40} className="text-slate-700 mb-3" />
                <p className="text-sm md:text-base mb-1 text-center">
                  Connect to the Realtime API to start the conversation.
                </p>
                <p className="text-[11px] text-slate-500 mb-3 text-center max-w-xs">
                  Your microphone audio will be transcribed, and the assistant will reply in
                  real time.
                </p>
                <button
                  onClick={connect}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-xs bg-indigo-600 hover:bg-indigo-700 text-white shadow shadow-indigo-500/30"
                >
                  <Mic size={14} /> Connect & Start
                </button>
              </div>
            ) : (
              <>
                {items.map(renderConversationItem)}

                {isConnected && items.length === 0 && (
                  <div className="h-full flex flex-col itemscenter justify-center text-slate-500">
                    <div className="flex justify-center mb-3">
                      <div
                        className={`w-14 h-14 md:w-16 md:h-16 rounded-full flex items-center justify-center transition-all ${
                          volume > 0.1 ? 'bg-indigo-500/20 scale-110' : 'bg-slate-800'
                        }`}
                      >
                        <Mic
                          size={28}
                          className={`transition-colors ${
                            isRecording && volume > 0.05 ? 'text-red-400' : 'text-slate-600'
                          }`}
                        />
                      </div>
                    </div>
                    <p className="text-sm text-center">
                      {isRecording
                        ? 'Speak into the microphone to begin.'
                        : 'Press the mic button or SPACEBAR to start.'}
                    </p>
                    <p className="text-[11px] text-slate-500 mt-1 text-center max-w-sm">
                      Your audio will be transcribed and the assistant will reply in real time.
                    </p>
                  </div>
                )}
              </>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Footer: status + input */}
          <div className="px-3 md:px-4 py-2 border-t border-slate-800 bg-slate-950">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2 text-[11px] md:text-xs text-slate-400">
                {isRecording && (
                  <div className="flex items-end h-3 mr-1">
                    {[...Array(5)].map((_, i) => (
                      <div
                        key={i}
                        className={`w-0.5 mx-0.5 rounded-sm transition-all duration-75 ${
                          volume > i * 0.2 ? 'bg-red-500' : 'bg-slate-700'
                        }`}
                        style={{
                          height: volume > i * 0.2 ? `${35 + i * 10}%` : '20%',
                        }}
                      />
                    ))}
                  </div>
                )}
                <span
                  className={
                    serverStatus === 'Listening...' && isRecording
                      ? 'text-red-400'
                      : serverStatus === 'Transcribing...'
                      ? 'text-indigo-400 animate-pulse'
                      : 'text-slate-400'
                  }
                >
                  {isConnected
                    ? isRecording
                      ? serverStatus === 'Listening...'
                        ? 'Listening…'
                        : serverStatus === 'Transcribing...'
                        ? 'Transcribing…'
                        : serverStatus === 'Processing...'
                        ? 'Processing…'
                        : serverStatus === 'Responding'
                        ? 'Generating response…'
                        : 'Type or speak (Spacebar to pause)'
                      : 'Mic paused (Spacebar to resume)'
                    : 'Not connected'}
                </span>
              </div>
              <div className="text-[10px] text-slate-500">
                {inputText.length}
                /2000
              </div>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendText()}
                placeholder={
                  isConnected ? 'Type your message…' : 'Connect first to start messaging…'
                }
                className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:ring-2 focus:ring-indigo-500 focus:outline-none placeholder:text-slate-500"
                disabled={!isConnected}
                maxLength={2000}
              />
              <button
                onClick={handleSendText}
                disabled={!isConnected || inputText.trim() === ''}
                className={`p-2.5 rounded-lg transition-all ${
                  isConnected && inputText.trim() !== ''
                    ? 'bg-indigo-600 hover:bg-indigo-700 text-white shadow shadow-indigo-500/30'
                    : 'bg-slate-800 text-slate-500'
                }`}
                title="Send Message"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
