export type SseEvent = {
  event: string;
  data: unknown;
};

export function parseSseFrame(frame: string): SseEvent | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) };
  } catch {
    return null;
  }
}

export class SseParser {
  private buffer = '';

  push(chunk: string): SseEvent[] {
    this.buffer += chunk;
    const events: SseEvent[] = [];
    let separatorIndex = this.findSeparator();
    while (separatorIndex >= 0) {
      const frame = this.buffer.slice(0, separatorIndex);
      this.buffer = this.buffer.slice(separatorIndex + this.separatorLength(separatorIndex));
      const parsed = parseSseFrame(frame);
      if (parsed) events.push(parsed);
      separatorIndex = this.findSeparator();
    }
    return events;
  }

  finish(): SseEvent[] {
    if (!this.buffer.trim()) return [];
    const parsed = parseSseFrame(this.buffer);
    this.buffer = '';
    return parsed ? [parsed] : [];
  }

  private findSeparator(): number {
    const lf = this.buffer.indexOf('\n\n');
    const crlf = this.buffer.indexOf('\r\n\r\n');
    if (lf < 0) return crlf;
    if (crlf < 0) return lf;
    return Math.min(lf, crlf);
  }

  private separatorLength(index: number): number {
    return this.buffer.slice(index, index + 4) === '\r\n\r\n' ? 4 : 2;
  }
}
