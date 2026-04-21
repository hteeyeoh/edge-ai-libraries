// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export enum MessageRole {
  Assistant = 'assistant',
  User = 'user',
  System = 'system',
}

export interface Message {
  role: MessageRole;
  content: string;
  time: number;
  conversationId?: string;
}

export interface ConversationRequest {
  conversationId: string;
  userPrompt: Message;
}

export interface Conversation {
  conversationId: string;
  title?: string;
  messages: Message[];
  responseStatus?: boolean;
}

export interface ConversationReducer {
  selectedConversationId: string;
  conversations: Conversation[];
  onGoingResults: { [conversationId: string]: string };
  files: string[];
  isGenerating: { [conversationId: string]: boolean };
  isWaitingForFirstToken: { [conversationId: string]: boolean };
  isUploading: boolean;
}
