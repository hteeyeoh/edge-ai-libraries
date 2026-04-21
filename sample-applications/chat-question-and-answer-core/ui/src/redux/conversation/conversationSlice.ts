// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { createSelector } from 'reselect';

import {
  ConversationReducer,
  ConversationRequest,
  Message,
  MessageRole,
} from './conversation.ts';
import client from '../../utils/client.ts';
import { CHAT_QNA_URL, DATA_PREP_URL } from '../../config.ts';
import {
  checkHealth,
  decodeEscapedBytes,
  getCurrentTimeStamp,
  getTitle,
  removeLastTagIfPresent,
  uuidv4,
} from '../../utils/util.ts';
import store, { RootState } from '../store.ts';
import {
  NotificationSeverity,
  notify,
} from '../../components/Notification/notify.ts';

const initialState: ConversationReducer = {
  conversations: [],
  selectedConversationId: '',
  onGoingResults: {},
  files: [],
  isGenerating: {},
  isWaitingForFirstToken: {},
  isUploading: false,
};

export const conversationSlice = createSlice({
  name: 'conversation',
  initialState,
  reducers: {
    logout: (state) => {
      state.conversations = [];
      state.selectedConversationId = '';
      state.onGoingResults = {};
      state.files = [];
      state.isGenerating = {};
      state.isWaitingForFirstToken = {};
      state.isUploading = false;
    },

    setOnGoingResultForConversation: (
      state,
      action: PayloadAction<{ conversationId: string; result: string }>,
    ) => {
      state.onGoingResults[action.payload.conversationId] = action.payload.result;
    },

    clearOnGoingResultForConversation: (state, action: PayloadAction<string>) => {
      delete state.onGoingResults[action.payload];
    },

    setIsGenerating: (
      state,
      action: PayloadAction<{ conversationId: string; isGenerating: boolean }>,
    ) => {
      const { conversationId, isGenerating } = action.payload;
      if (isGenerating) {
        state.isGenerating[conversationId] = true;
      } else {
        delete state.isGenerating[conversationId];
      }
    },

    setIsWaitingForFirstToken: (
      state,
      action: PayloadAction<{ conversationId: string; isWaiting: boolean }>,
    ) => {
      const { conversationId, isWaiting } = action.payload;
      if (isWaiting) {
        state.isWaitingForFirstToken[conversationId] = true;
      } else {
        delete state.isWaitingForFirstToken[conversationId];
      }
    },

    addMessageToMessages: (state, action: PayloadAction<Message>) => {
      const targetConversationId =
        action.payload.conversationId || state.selectedConversationId;
      const selectedConversation = state.conversations.find((conversation) => {
        return conversation.conversationId === targetConversationId;
      });
      selectedConversation?.messages?.push(action.payload);
    },

    newConversation: (state) => {
      state.selectedConversationId = '';
    },

    deleteConversation: (state, action: PayloadAction<string>) => {
      const conversationId = action.payload;
      const conversationIndex = state.conversations.findIndex(
        (conversation) => conversation.conversationId === conversationId,
      );
      if (conversationIndex !== -1) {
        state.conversations.splice(conversationIndex, 1);
        if (state.selectedConversationId === conversationId) {
          state.selectedConversationId = '';
        }
        if (state.conversations.length === 0) {
          state.selectedConversationId = '';
        }
      }
      delete state.onGoingResults[conversationId];
      delete state.isGenerating[conversationId];
      delete state.isWaitingForFirstToken[conversationId];
    },

    updateConversationTitle: (
      state,
      action: PayloadAction<{ id: string; updatedTitle: string }>,
    ) => {
      const selectedConversation = state.conversations.find(
        (conversation) => conversation.conversationId === action.payload.id,
      );
      if (selectedConversation)
        selectedConversation.title = action.payload.updatedTitle;
    },

    createNewConversation: (
      state,
      action: PayloadAction<{ title: string; id: string; message: Message }>,
    ) => {
      state.conversations.unshift({
        title: action.payload.title,
        conversationId: action.payload.id,
        messages: [action.payload.message],
        responseStatus: false,
      });
    },

    setSelectedConversationId: (state, action: PayloadAction<string>) => {
      state.selectedConversationId = action.payload;
    },

    setResponseStatus: (state, action: PayloadAction<boolean>) => {
      const selectedConversation = state.conversations.find(
        (conversation) =>
          conversation.conversationId === state.selectedConversationId,
      );
      if (selectedConversation) {
        selectedConversation.responseStatus = action.payload;
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchInitialFiles.fulfilled, (state, action) => {
      state.files = action.payload.data;
    });
    builder.addCase(fetchInitialFiles.rejected, (state) => {
      state.files = [];
    });
    builder.addCase(uploadFile.pending, (state) => {
      state.isUploading = true;
    });
    builder.addCase(uploadFile.fulfilled, (state) => {
      state.isUploading = false;
    });
    builder.addCase(uploadFile.rejected, (state) => {
      state.isUploading = false;
    });
    builder.addCase(removeFile.fulfilled, (state, action) => {
      const index = state.files.findIndex((file) => file === action.payload);
      if (index !== -1) {
        state.files.splice(index, 1);
      }
    });
    builder.addCase(removeFile.rejected, () => { });
    builder.addCase(removeAllFiles.fulfilled, (state, action) => {
      state.files = action.payload;
    });
    builder.addCase(removeAllFiles.rejected, () => { });
  },
});

const handleConnectionError = async (message?: string) => {
  const healthStatus = await checkHealth();

  if (healthStatus.status === 503) {
    notify(
      message ||
      'The backend service is starting up. Please try again in a few moments.',
      NotificationSeverity.ERROR,
    );
  }
};

export const fetchInitialFiles = createAsyncThunk(
  'conversation/fetchInitialFiles',
  async (_, { rejectWithValue }) => {
    try {
      const response = await client.get(DATA_PREP_URL);
      if (response.status === 200) {
        const validFiles: string[] = response.data.metadata.documents;
        return { data: validFiles, status: response.status };
      } else {
        throw new Error(`Request failed with status code ${response.status}`);
      }
    } catch (error) {
      if (client.isAxiosError(error) && error.response) {
        return rejectWithValue({
          status: error.status,
          message: error.message || 'Failed to fetch files',
        });
      } else {
        return rejectWithValue({
          status: 500,
          message: (error as Error).message || 'An unknown error occured',
        });
      }
    }
  },
);

export const uploadFile = createAsyncThunk(
  'conversation/uploadFile',
  async ({ file }: { file: File }, { rejectWithValue }) => {
    try {
      const body = new FormData();
      body.append('files', file);

      await handleConnectionError();

      const response = await client.post(DATA_PREP_URL, body);

      if (response.status === 200) {
        return { data: response.data, status: response.status };
      } else {
        throw new Error(`Request failed with status code ${response.status}`);
      }
    } catch (error) {
      if (client.isAxiosError(error) && error.response) {
        return rejectWithValue({
          status: error.status,
          message: error.message || 'Failed to upload the file',
        });
      } else {
        return rejectWithValue({
          status: 500,
          message: (error as Error).message || 'An unknown error occured',
        });
      }
    }
  },
);

export const removeFile = createAsyncThunk(
  'conversation/removeFile',
  async (
    { fileName, deleteAll = false }: { fileName: string; deleteAll?: boolean },
    { getState, rejectWithValue },
  ) => {
    try {
      const state = getState() as RootState;
      const file = state.conversations.files.find((file) => file === fileName);

      if (!file) {
        throw new Error('File not found');
      }

      await handleConnectionError();

      const response = await client.delete(
        `${DATA_PREP_URL}?document=${encodeURIComponent(fileName)}&delete_all=${deleteAll}`,
      );

      if (response.status === 204) {
        return fileName;
      } else {
        throw new Error(`Request failed with status code ${response.status}`);
      }
    } catch (error) {
      if (client.isAxiosError(error) && error.response) {
        return rejectWithValue({
          status: error.status,
          message: error.message || 'Failed to delete the file',
        });
      } else {
        return rejectWithValue({
          status: 500,
          message: (error as Error).message || 'An unknown error occurred',
        });
      }
    }
  },
);

export const removeAllFiles = createAsyncThunk(
  'conversation/removeAllFiles',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as RootState;

      if (state.conversations.files.length === 0) {
        throw new Error('No files to delete');
      }

      await handleConnectionError();

      const response = await client.delete(
        `${DATA_PREP_URL}?delete_all=${true}`,
      );
      if (response.status === 204) {
        return [];
      } else {
        throw new Error(`Request failed with status code ${response.status}`);
      }
    } catch (error) {
      if (client.isAxiosError(error) && error.response) {
        return rejectWithValue({
          status: error.status,
          message: error.message || 'Failed to delete the file(s)',
        });
      } else {
        return rejectWithValue({
          status: 500,
          message: (error as Error).message || 'An unknown error occurred',
        });
      }
    }
  },
);

export const doConversation = (conversationRequest: ConversationRequest) => {
  const { userPrompt } = conversationRequest;
  const inputConversationId = conversationRequest.conversationId;
  let activeConversationId = inputConversationId;

  if (!activeConversationId) {
    activeConversationId = uuidv4();
    store.dispatch(
      createNewConversation({
        title: getTitle(userPrompt.content),
        id: activeConversationId,
        message: userPrompt,
      }),
    );
    store.dispatch(setSelectedConversationId(activeConversationId));
  } else {
    store.dispatch(
      addMessageToMessages({
        ...userPrompt,
        conversationId: activeConversationId,
      }),
    );
  }

  const currentState = store.getState();
  const selectedConversation = currentState.conversations.conversations.find(
    (conversation) => conversation.conversationId === activeConversationId,
  );
  const conversationMessages = (selectedConversation?.messages || []).map(
    (message) => ({ role: message.role, content: message.content }),
  );

  const body = {
    conversation_messages: conversationMessages,
    stream: true,
  };

  store.dispatch(
    setIsGenerating({ conversationId: activeConversationId, isGenerating: true }),
  );
  store.dispatch(
    setIsWaitingForFirstToken({
      conversationId: activeConversationId,
      isWaiting: true,
    }),
  );
  store.dispatch(setResponseStatus(false));

  handleConnectionError().catch(console.error);

  let result: string = '';
  let firstTokenReceived = false;
  try {
    fetchEventSource(CHAT_QNA_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      openWhenHidden: true,
      async onopen(response) {
        if (response.ok) {
          return;
        } else if (
          response.status >= 400 &&
          response.status < 500 &&
          response.status !== 429
        ) {
          const e = await response.json();
          console.log(e);
          throw Error(e.error.message);
        } else {
          console.log('error', response);
        }
      },
      onmessage(msg) {
        if (msg?.data !== '[DONE]') {
          try {
            if (msg.data) {
              const chunk = msg.data.includes('\\x')
                ? decodeEscapedBytes(msg.data)
                : msg.data;
              result += chunk;
              result = removeLastTagIfPresent(result);
              if (result) {
                if (!firstTokenReceived) {
                  firstTokenReceived = true;
                  store.dispatch(
                    setIsWaitingForFirstToken({
                      conversationId: activeConversationId,
                      isWaiting: false,
                    }),
                  );
                }
                store.dispatch(
                  setOnGoingResultForConversation({
                    conversationId: activeConversationId,
                    result,
                  }),
                );
              }
            }
          } catch (e) {
            console.error('Error parsing message:', e);
          }
        }
      },
      onerror(err) {
        console.log('error', err);
        store.dispatch(
          setIsGenerating({
            conversationId: activeConversationId,
            isGenerating: false,
          }),
        );
        store.dispatch(clearOnGoingResultForConversation(activeConversationId));
        store.dispatch(
          setIsWaitingForFirstToken({
            conversationId: activeConversationId,
            isWaiting: false,
          }),
        );

        throw err;
      },
      onclose() {
        store.dispatch(clearOnGoingResultForConversation(activeConversationId));
        store.dispatch(
          setIsGenerating({
            conversationId: activeConversationId,
            isGenerating: false,
          }),
        );
        store.dispatch(
          setIsWaitingForFirstToken({
            conversationId: activeConversationId,
            isWaiting: false,
          }),
        );
        store.dispatch(setResponseStatus(true));
        store.dispatch(
          addMessageToMessages({
            role: MessageRole.Assistant,
            content: result,
            time: getCurrentTimeStamp(),
            conversationId: activeConversationId,
          }),
        );
      },
    });
  } catch (err) {
    console.log(err);
  }
};

export const {
  logout,
  setOnGoingResultForConversation,
  clearOnGoingResultForConversation,
  setIsGenerating,
  setIsWaitingForFirstToken,
  newConversation,
  deleteConversation,
  updateConversationTitle,
  addMessageToMessages,
  setSelectedConversationId,
  setResponseStatus,
  createNewConversation,
} = conversationSlice.actions;

const selectConversationState = (state: RootState) => state.conversations;

export const conversationSelector = createSelector(
  [selectConversationState],
  (conversationState) => ({
    files: conversationState?.files || [],
    conversations: conversationState?.conversations || [],
    selectedConversationId: conversationState?.selectedConversationId || '',
    onGoingResults: conversationState?.onGoingResults || {},
    isGenerating: conversationState?.isGenerating || {},
    isWaitingForFirstToken: conversationState?.isWaitingForFirstToken || {},
    isUploading: conversationState?.isUploading || false,
  }),
);

export default conversationSlice.reducer;
