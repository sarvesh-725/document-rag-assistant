export type DocumentStatus = 'PROCESSING' | 'READY' | 'FAILED' | 'DELETING';

export type Document = {
  document_id: string;
  display_name: string;
  original_filename: string;
  created_at: string;
  status: DocumentStatus;
  duplicate_index: number;
};
