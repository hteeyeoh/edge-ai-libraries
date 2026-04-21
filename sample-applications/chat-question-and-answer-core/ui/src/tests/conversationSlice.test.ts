// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { configureStore } from '@reduxjs/toolkit';
import { describe, it, expect } from 'vitest';

import conversationReducer, {
    logout,
    setOnGoingResultForConversation,
    clearOnGoingResultForConversation,
    setIsGenerating,
    setIsWaitingForFirstToken,
    addMessageToMessages,
    newConversation,
    deleteConversation,
    updateConversationTitle,
    createNewConversation,
    setSelectedConversationId,
} from '../redux/conversation/conversationSlice.ts';
import {
    ConversationReducer,
    Message,
    MessageRole,
} from '../redux/conversation/conversation.ts';

describe('conversationSlice reducers', () => {
    const initialState: ConversationReducer = {
        conversations: [],
        selectedConversationId: '',
        onGoingResults: {},
        files: [],
        isGenerating: {},
        isWaitingForFirstToken: {},
        isUploading: false,
    };

    it('returns initial state', () => {
        expect(conversationReducer(undefined, { type: 'unknown' })).toEqual(
            initialState,
        );
    });

    it('handles setOnGoingResultForConversation and clear', () => {
        const setState = conversationReducer(
            initialState,
            setOnGoingResultForConversation({
                conversationId: 'c1',
                result: 'partial',
            }),
        );
        expect(setState.onGoingResults.c1).toBe('partial');

        const clearState = conversationReducer(
            setState,
            clearOnGoingResultForConversation('c1'),
        );
        expect(clearState.onGoingResults.c1).toBeUndefined();
    });

    it('handles setIsGenerating and setIsWaitingForFirstToken', () => {
        const generating = conversationReducer(
            initialState,
            setIsGenerating({ conversationId: 'c1', isGenerating: true }),
        );
        expect(generating.isGenerating.c1).toBe(true);

        const waiting = conversationReducer(
            generating,
            setIsWaitingForFirstToken({ conversationId: 'c1', isWaiting: true }),
        );
        expect(waiting.isWaitingForFirstToken.c1).toBe(true);

        const cleared = conversationReducer(
            waiting,
            setIsGenerating({ conversationId: 'c1', isGenerating: false }),
        );
        expect(cleared.isGenerating.c1).toBeUndefined();
    });

    it('creates conversation, appends message, updates title, selects and deletes', () => {
        const userMessage: Message = {
            role: MessageRole.User,
            content: 'Hello',
            time: 100,
            conversationId: 'c1',
        };

        const created = conversationReducer(
            initialState,
            createNewConversation({ title: 'Chat 1', id: 'c1', message: userMessage }),
        );
        expect(created.conversations).toHaveLength(1);

        const selected = conversationReducer(created, setSelectedConversationId('c1'));
        expect(selected.selectedConversationId).toBe('c1');

        const assistantMessage: Message = {
            role: MessageRole.Assistant,
            content: 'Hi',
            time: 101,
            conversationId: 'c1',
        };
        const withReply = conversationReducer(
            selected,
            addMessageToMessages(assistantMessage),
        );
        expect(withReply.conversations[0].messages).toHaveLength(2);

        const renamed = conversationReducer(
            withReply,
            updateConversationTitle({ id: 'c1', updatedTitle: 'Renamed' }),
        );
        expect(renamed.conversations[0].title).toBe('Renamed');

        const deleted = conversationReducer(renamed, deleteConversation('c1'));
        expect(deleted.conversations).toHaveLength(0);
        expect(deleted.selectedConversationId).toBe('');
    });

    it('handles newConversation and logout', () => {
        const stateWithSelection = {
            ...initialState,
            selectedConversationId: 'c1',
        };
        const resetSelection = conversationReducer(stateWithSelection, newConversation());
        expect(resetSelection.selectedConversationId).toBe('');

        const loggedOut = conversationReducer(resetSelection, logout());
        expect(loggedOut).toEqual(initialState);
    });

    it('is compatible with store setup', () => {
        const store = configureStore({
            reducer: { conversation: conversationReducer },
            preloadedState: { conversation: initialState },
        });
        expect(store.getState().conversation).toEqual(initialState);
    });
});
