export type DocumentStatus = 'PROCESSING' | 'READY' | 'FAILED' | 'DELETING';

export type Document = {
  document_id: string;
  display_name: string;
  original_filename: string;
  created_at: string;
  status: DocumentStatus;
  duplicate_index: number;
};

export type Source = {
  source_id: string;
  document_id: string;
  version_id?: string;
  display_name: string;
  page?: number;
  section?: string;
  chunk_id?: string;
};

export type ChatMessage = {
  message_id?: string;
  sequence_number?: number;
  client_request_id?: string;
  status?: string;
  role: 'user' | 'assistant';
  text: string;
  bound_document_ids?: string[];
  sources?: Source[];
};

export type SessionSummary = {
  id: string;
  title: string | null;
};
