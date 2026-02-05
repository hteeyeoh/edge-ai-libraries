// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export enum MessageRole {
  Assistant = 'assistant',
  User = 'user',
  System = 'system',
}

export interface FrameMetadata {
  frame_data: string; // base64 encoded image
  frame_format: string; // e.g., 'BGRA'
  frame_height: number;
  frame_width: number;
  frame_id: number;
}

export interface FrameSource {
  metadata: FrameMetadata;
  preview: string; // text preview/caption
}

export interface Message {
  role: MessageRole;
  content: string;
  time: number;
  conversationId: string;
  frames?: FrameSource[]; // optional frame data
}

export interface ConversationRequest {
  userPrompt: Message;
}

export interface Conversation {
  conversationId: string;
  title?: string;
  messages: Message[];
  responseStatus: boolean;
}

export interface ConversationReducer {
  selectedConversationId: string;
  conversations: Conversation[];
  onGoingResult: string;
  files: string[];
  isGenerating: boolean;
}
