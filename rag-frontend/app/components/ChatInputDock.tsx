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
  selectedFiles: string[];
  setSelectedFiles: React.Dispatch<React.SetStateAction<string[]>>;
  onDelete: (document_id: string) => Promise<void>;
  globalFiles: FileItem[];
  fetchGlobalFiles: () => Promise<void>;
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
  selectedFiles,
  setSelectedFiles,
  onDelete,
  globalFiles,
  fetchGlobalFiles
}: ChatInputDockProps) {
  const [showOverlay, setShowOverlay] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (showOverlay) {
      fetchGlobalFiles();
    }
  }, [showOverlay, sessionFiles, fetchGlobalFiles]);

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
  globalFiles.forEach(f => allUniqueFilesMap.set(f.document_id, f));
  sessionFiles.forEach(f => allUniqueFilesMap.set(f.document_id, f));

  const allUniqueFiles = Array.from(allUniqueFilesMap.values())
    .filter((f) => f.display_name.toLowerCase().includes(searchQuery.toLowerCase()));

  const isActive = (file: FileItem) => {
    return selectedFiles.includes(file.document_id);
  };

  const handleToggleFile = async (file: FileItem) => {
    const active = isActive(file);
    if (active) {
      setSelectedFiles(prev => prev.filter(id => id !== file.document_id));
    } else {
      setSelectedFiles(prev => [...prev, file.document_id]);
    }
  };

  const unselectedGlobalFiles = globalFiles
    .filter((gf) => !selectedFiles.includes(gf.document_id));

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true);
    setUploadError(null);

    const isAlreadyPresent = sessionFiles.some((sf) => sf.original_filename === file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents`,
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
      const data = await res.json();

      setShowOverlay(false);
      await fetchFiles();
      if (data.document_id && !selectedFiles.includes(data.document_id)) {
        setSelectedFiles(prev => [...prev, data.document_id]);
      }
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
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/documents`,
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
      const data = await res.json();

      setShowOverlay(false);
      await fetchFiles();
      if (data.document_id && !selectedFiles.includes(data.document_id)) {
        setSelectedFiles(prev => [...prev, data.document_id]);
      }
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

                return (
                  <div
                    key={file.document_id}
                    onClick={() => {
                      if (!isProcessing) {
                        handleToggleFile(file);
                      }
                    }}
                    className={`flex items-center justify-between p-2 rounded-lg border transition-all duration-150 ${
                      active
                        ? 'bg-slate-800/80 border-indigo-500/30'
                        : 'bg-slate-950/30 border-slate-800/40 hover:border-slate-700/30'
                    } ${isProcessing ? 'cursor-not-allowed opacity-80' : 'cursor-pointer'} group`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <input
                        type="checkbox"
                        checked={active}
                        disabled={isProcessing}
                        onChange={() => {
                          if (!isProcessing) {
                            handleToggleFile(file);
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-indigo-600 focus:ring-offset-slate-900 cursor-pointer disabled:cursor-not-allowed"
                      />
                      <span className="text-xs text-slate-300 font-medium truncate max-w-[220px]">
                        {file.display_name}
                      </span>
                    </div>

                    <div className="text-[10px] flex items-center gap-3 font-semibold">
                      {isProcessing ? (
                        <div className="flex items-center gap-1 text-indigo-400">
                          <Loader2 size={10} className="animate-spin" />
                          <span>Ingesting</span>
                        </div>
                      ) : isFailed ? (
                        <span className="text-red-400">Ingestion Failed</span>
                      ) : null}
                      <button
                        onClick={async (e) => {
                          e.stopPropagation();
                          if (confirm(`Are you sure you want to permanently delete ${file.display_name}?`)) {
                            await onDelete(file.document_id);
                            fetchGlobalFiles();
                          }
                        }}
                        className="text-slate-500 hover:text-red-400 p-1 rounded transition-colors opacity-0 group-hover:opacity-100"
                        title="Delete file permanently"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>
                      </button>
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
                disabled={queryLoading}
                rows={1}
                value={inputQuestion}
                onChange={(e) => setInputQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (inputQuestion.trim() && !queryLoading) {
                      onSubmit(e);
                    }
                  }
                }}
                placeholder="Ask a question (RAG active if files selected, else standard chat)..."
                className="w-full bg-transparent border-0 resize-none text-sm text-slate-100 placeholder-slate-500 focus:ring-0 focus:outline-none min-h-[36px] py-2 max-h-[160px] pr-2 scrollbar-thin scrollbar-thumb-slate-800"
              />
            </div>

            {}
            <button
              type="submit"
              disabled={queryLoading || !inputQuestion.trim()}
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
