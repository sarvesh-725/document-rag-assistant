'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Paperclip, Send, Loader2, Search, UploadCloud, CheckCircle } from 'lucide-react';
import { FileItem } from './SelectedFilesBar';

interface ChatInputDockProps {
  token: string;
  sessionId: string;
  inputQuestion: string;
  setInputQuestion: (val: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  queryLoading: boolean;
  sessionFiles: FileItem[];
  fetchFiles: () => Promise<void>;
  onBind: (filename: string) => Promise<void>;
  onUnbind: (filename: string) => Promise<void>;
}

export default function ChatInputDock({
  token,
  sessionId,
  inputQuestion,
  setInputQuestion,
  onSubmit,
  queryLoading,
  sessionFiles,
  fetchFiles,
  onBind,
  onUnbind
}: ChatInputDockProps) {
  const [showOverlay, setShowOverlay] = useState(false);
  const [globalFiles, setGlobalFiles] = useState<FileItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  const fetchGlobalFiles = async () => {
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents/global`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        setGlobalFiles(data);
      }
    } catch (err) {
      console.error('Error fetching global files:', err);
    }
  };

  useEffect(() => {
    if (showOverlay) {
      fetchGlobalFiles();
    }
  }, [showOverlay, sessionFiles]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (overlayRef.current && !overlayRef.current.contains(event.target as Node)) {
        setShowOverlay(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const allUniqueFilesMap = new Map<string, FileItem>();
  globalFiles.forEach(f => allUniqueFilesMap.set(f.filename, f));
  sessionFiles.forEach(f => allUniqueFilesMap.set(f.filename, f));

  const allUniqueFiles = Array.from(allUniqueFilesMap.values())
    .filter((f) => f.filename.toLowerCase().includes(searchQuery.toLowerCase()));

  const isActive = (file: FileItem) => {
    const sessionFile = sessionFiles.find(sf => sf.filename === file.filename);
    return sessionFile ? (sessionFile.is_committed || sessionFile.just_uploaded || sessionFile.status === 'processing') : false;
  };

  const handleToggleFile = async (file: FileItem) => {
    const active = isActive(file);
    if (active) {
      if (file.just_uploaded) {
        return; // Disable unchecking just uploaded files in checklist (they use Undo button)
      }
      await onUnbind(file.filename);
    } else {
      await onBind(file.filename);
    }
    fetchGlobalFiles();
  };



  const unselectedGlobalFiles = globalFiles
    .filter((gf) => !sessionFiles.some((sf) => sf.filename === gf.filename && (sf.is_committed || sf.just_uploaded || sf.status === 'processing')));

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true);
    setUploadError(null);

    const isAlreadyPresent = sessionFiles.some((sf) => sf.filename === file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents/upload?session_id=${sessionId}`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      setShowOverlay(false);
      await fetchFiles();
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (!e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
    const file = e.dataTransfer.files[0];
    
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'pdf' && ext !== 'txt') {
      setUploadError('Invalid file type. Only PDF and TXT files are accepted.');
      return;
    }

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents/upload?session_id=${sessionId}`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      setShowOverlay(false);
      await fetchFiles();
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="relative w-full">
      {}
      {showOverlay && (
        <div
          ref={overlayRef}
          className="absolute bottom-full left-0 mb-3 w-[450px] bg-slate-900/95 border border-slate-800 rounded-2xl shadow-2xl p-4 space-y-4 backdrop-blur-xl z-50 transition-all duration-200 animate-in fade-in slide-in-from-bottom-2"
        >
          <div className="flex justify-between items-center pb-2 border-b border-slate-800">
            <h4 className="font-bold text-xs uppercase tracking-wider text-indigo-400">Context Assets Panel</h4>
            <button
              onClick={() => setShowOverlay(false)}
              className="text-xs text-slate-400 hover:text-slate-200 cursor-pointer"
            >
              Close
            </button>
          </div>

          {}
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-800 hover:border-indigo-500/40 bg-slate-950/40 hover:bg-slate-950/60 rounded-xl p-6 flex flex-col items-center justify-center gap-2 cursor-pointer transition-all duration-200 group"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf,.txt"
              className="hidden"
            />
            {uploading ? (
              <Loader2 size={24} className="animate-spin text-indigo-500" />
            ) : (
              <UploadCloud size={24} className="text-slate-500 group-hover:text-indigo-400 transition-colors" />
            )}
            <span className="text-xs font-semibold text-slate-300">
              {uploading ? 'Processing upload...' : 'Upload PDF or TXT File'}
            </span>
            <span className="text-[10px] text-slate-500">Drag & drop or click to browse</span>
          </div>

          {uploadError && (
            <div className="text-[10px] text-red-400 bg-red-950/20 border border-red-500/20 p-2 rounded-lg">
              {uploadError}
            </div>
          )}

          {}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Historical Files</span>
            </div>

            {}
            <div className="relative">
              <Search size={14} className="absolute left-3 top-2.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search global indexes..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-slate-950/80 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-xs focus:border-indigo-500 focus:outline-none transition-colors text-slate-200"
              />
            </div>

            {}
            <div className="max-h-[160px] overflow-y-auto space-y-1.5 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent pr-1">
              {allUniqueFiles.map((file) => {
                const active = isActive(file);
                const isProcessing = file.status === 'processing';
                const isFailed = file.status === 'failed';
                const isJustUploaded = file.just_uploaded;

                return (
                  <div
                    key={file.filename}
                    onClick={() => {
                      if (!isProcessing && !isJustUploaded) {
                        handleToggleFile(file);
                      }
                    }}
                    className={`flex items-center justify-between p-2 rounded-lg border transition-all duration-150 ${
                      active
                        ? 'bg-slate-800/80 border-indigo-500/30'
                        : 'bg-slate-950/30 border-slate-800/40 hover:border-slate-700/30'
                    } ${isProcessing || isJustUploaded ? 'cursor-not-allowed opacity-80' : 'cursor-pointer'}`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <input
                        type="checkbox"
                        checked={active}
                        disabled={isProcessing || isJustUploaded}
                        onChange={() => {}} // Handled by parent div onClick
                        className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-indigo-600 focus:ring-offset-slate-900 cursor-pointer disabled:cursor-not-allowed"
                      />
                      <span className="text-xs text-slate-300 font-medium truncate max-w-[240px]">
                        {file.filename}
                      </span>
                    </div>

                    <div className="text-[10px] flex items-center gap-1.5 font-semibold">
                      {isProcessing ? (
                        <div className="flex items-center gap-1 text-indigo-400">
                          <Loader2 size={10} className="animate-spin" />
                          <span>Ingesting</span>
                        </div>
                      ) : isFailed ? (
                        <span className="text-red-400">Ingestion Failed</span>
                      ) : isJustUploaded ? (
                        <span className="text-indigo-400 text-[9px] uppercase tracking-wider bg-indigo-500/10 px-1.5 py-0.5 rounded">
                          Just Uploaded
                        </span>
                      ) : active ? (
                        <span className="text-emerald-400 text-[9px] uppercase tracking-wider bg-emerald-500/10 px-1.5 py-0.5 rounded">
                          Active
                        </span>
                      ) : (
                        <span className="text-slate-500 group-hover:text-slate-400 transition-colors text-[9px] uppercase tracking-wider font-semibold">
                          Inactive
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}

              {allUniqueFiles.length === 0 && (
                <p className="text-[11px] text-slate-500 italic text-center py-4">
                  No documents found in global index.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {}
      {(() => {
        const hasActiveFiles = sessionFiles.some(
          (f) => f.is_committed || f.just_uploaded || f.status === 'processing'
        );

        return (
          <form onSubmit={onSubmit} className="flex gap-2 items-end bg-slate-900 border border-slate-800 rounded-2xl p-2 focus-within:border-indigo-500/50 transition-colors shadow-xl">
            <div className="flex-1 flex items-center relative pl-10">
              {}
              <button
                type="button"
                onClick={() => setShowOverlay(!showOverlay)}
                className="absolute left-2 bottom-2 text-slate-400 hover:text-slate-200 p-2 rounded-xl hover:bg-slate-800 transition-all cursor-pointer"
                title="Attach documents"
              >
                <Paperclip size={18} />
              </button>

              {}
              <textarea
                disabled={queryLoading || !hasActiveFiles}
                rows={1}
                value={inputQuestion}
                onChange={(e) => setInputQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (inputQuestion.trim() && !queryLoading && hasActiveFiles) {
                      onSubmit(e);
                    }
                  }
                }}
                placeholder={
                  hasActiveFiles
                    ? "Ask a question against active documents context..."
                    : "Attach or select a document to start querying..."
                }
                className={`w-full bg-transparent border-0 resize-none text-sm text-slate-100 placeholder-slate-500 focus:ring-0 focus:outline-none min-h-[36px] py-2 max-h-[160px] pr-2 scrollbar-thin scrollbar-thumb-slate-800 ${
                  !hasActiveFiles ? "cursor-not-allowed text-slate-500" : ""
                }`}
              />
            </div>

            {}
            <button
              type="submit"
              disabled={queryLoading || !inputQuestion.trim() || !hasActiveFiles}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white p-3 rounded-xl transition-all shrink-0 shadow-sm flex items-center justify-center h-10 w-10 cursor-pointer"
            >
              {queryLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Send size={16} />
              )}
            </button>
          </form>
        );
      })()}
    </div>
  );
}
