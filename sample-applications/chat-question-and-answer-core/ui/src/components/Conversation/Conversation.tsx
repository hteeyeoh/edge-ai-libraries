// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState, type FC, type KeyboardEvent } from 'react';
import { ActionIcon, Anchor, Group, Paper, Stack, Text, Title } from '@mantine/core';
import { IconFilePlus, IconMessagePlus } from '@tabler/icons-react';

import { useAppDispatch, useAppSelector } from '../../redux/store.ts';
import {
  doConversation,
  conversationSelector,
  fetchInitialFiles,
  newConversation,
} from '../../redux/conversation/conversationSlice.ts';
import { Message, MessageRole } from '../../redux/conversation/conversation.ts';
import ConversationMessage from './ConversationMessage.tsx';
import Textarea from '../Textarea/Textarea.tsx';
import ConversationSideBar from './ConversationSideBar.tsx';
import DataSource from '../Drawer/DataSource.tsx';
import { fetchModelName, getCurrentTimeStamp } from '../../utils/util.ts';

interface ConversationProps {
  title?: string;
}

const Conversation: FC<ConversationProps> = ({ title = 'ChatQnA' }) => {
  const [modelName, setModelName] = useState<string>('');
  const [hasLLMResponse, setHasLLMResponse] = useState<boolean>(false);
  const [fileUploadOpened, setFileUploadOpened] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>('');

  const dispatch = useAppDispatch();
  const {
    conversations,
    onGoingResults,
    selectedConversationId,
    isGenerating,
  } = useAppSelector(conversationSelector);

  const selectedConversation = conversations.find(
    (conversation) => conversation.conversationId === selectedConversationId,
  );

  const selectedConversationTitle = selectedConversation?.title || 'New conversation';
  const { responseStatus = false } = selectedConversation || {};

  const scrollViewport = useRef<HTMLDivElement>(null);

  const loadModelName = async () => {
    const response = await fetchModelName();
    if (response.status === 200) {
      setModelName(response.llmModel);
    }
  };

  const scrollToBottom = () => {
    scrollViewport.current?.scrollTo({
      top: scrollViewport.current.scrollHeight,
      behavior: 'smooth',
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [onGoingResults?.[selectedConversationId], selectedConversation?.messages]);

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        await dispatch(fetchInitialFiles()).unwrap();
      } catch {
        console.log('Failed to fetch files');
      }
    };

    fetchFiles();
  }, [dispatch]);

  useEffect(() => {
    void loadModelName();
  }, []);

  useEffect(() => {
    const hasAssistantReply =
      selectedConversation?.messages?.some(
        (message) => message.role === MessageRole.Assistant,
      ) || false;
    setHasLLMResponse(hasAssistantReply);
  }, [selectedConversation?.messages]);

  const handleNewConversation = () => {
    dispatch(newConversation());
    setPrompt('');
  };

  const handleSubmit = () => {
    if (!prompt.trim()) {
      return;
    }

    void loadModelName();

    const userPrompt: Message = {
      role: MessageRole.User,
      content: prompt.trim(),
      time: getCurrentTimeStamp(),
      conversationId: selectedConversationId || '',
    };
    doConversation({ conversationId: selectedConversationId || '', userPrompt });
    setPrompt('');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  const LLM_MODEL_URL: string = `https://huggingface.co/${modelName}`;

  return (
    <Group align='stretch' gap={0} h='100vh' data-testid='conversation-container'>
      <ConversationSideBar title={title} />
      <Stack flex={1} gap={0}>
        <Paper
          radius={0}
          p={0}
          withBorder
          style={{
            backgroundColor: 'var(--mantine-color-blue-6)',
            color: 'var(--color-white)',
            height: '64px',
            padding: '0 36px 0 16px',
            boxSizing: 'border-box',
          }}
        >
          <Group justify='space-between' style={{ width: '100%', height: '100%' }}>
            <Title order={4}>{selectedConversationTitle}</Title>
            <Group style={{ marginRight: '12px' }}>
              <ActionIcon
                variant='default'
                size='lg'
                radius='md'
                onClick={handleNewConversation}
                disabled={!!(selectedConversationId && isGenerating[selectedConversationId])}
                aria-label='New conversation'
              >
                <IconMessagePlus size={18} />
              </ActionIcon>
              <ActionIcon
                variant='default'
                size='lg'
                radius='md'
                onClick={() => setFileUploadOpened(true)}
                aria-label='Manage context'
              >
                <IconFilePlus size={18} />
              </ActionIcon>
            </Group>
          </Group>
        </Paper>

        <Stack flex={1} p='md' gap='sm' style={{ overflowY: 'auto' }} ref={scrollViewport}>
          {!selectedConversation ? (
            <Text c='dimmed'>Start by asking a question.</Text>
          ) : null}

          {selectedConversation?.messages.map((message, index) => (
            <ConversationMessage
              key={index}
              date={message.time * 1000}
              human={message.role === MessageRole.User}
              message={message.content}
            />
          ))}

          {(selectedConversationId &&
            (isGenerating[selectedConversationId] || onGoingResults[selectedConversationId])) ? (
            <ConversationMessage
              key={`ongoing-${selectedConversationId}`}
              date={Date.now()}
              human={false}
              message={onGoingResults[selectedConversationId] || ''}
              showBlinkingIndicator={!!isGenerating[selectedConversationId]}
            />
          ) : null}

          {(responseStatus || hasLLMResponse) && modelName ? (
            <Group justify='flex-end'>
              <Anchor href={LLM_MODEL_URL} target='_blank' size='sm'>
                {modelName}
              </Anchor>
            </Group>
          ) : null}
        </Stack>

        <Paper radius={0} p='md' withBorder>
          <Textarea
            rows={2}
            setModelName={setModelName}
            value={prompt}
            onChange={setPrompt}
            onSubmit={handleSubmit}
            onKeyDown={handleKeyDown}
          />
        </Paper>
      </Stack>

      <DataSource
        opened={fileUploadOpened}
        onClose={() => setFileUploadOpened(false)}
      />
    </Group>
  );
};

export default Conversation;
