// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  type ChangeEvent,
  type FC,
  type KeyboardEventHandler,
  useEffect,
  useState,
  useRef,
  KeyboardEvent,
  Dispatch,
  SetStateAction,
} from 'react';
import { ActionIcon, Group, Textarea as MantineTextarea } from '@mantine/core';
import { IconArrowRight } from '@tabler/icons-react';
import { useTranslation } from 'react-i18next';

import { Message, MessageRole } from '../../redux/conversation/conversation.ts';
import { fetchModelName, getCurrentTimeStamp } from '../../utils/util.ts';
import {
  conversationSelector,
  doConversation,
} from '../../redux/conversation/conversationSlice.ts';
import { NotificationSeverity, notify } from '../Notification/notify.ts';
import { useAppSelector } from '../../redux/store.ts';

interface TextareaProps {
  rows?: number;
  setModelName: Dispatch<SetStateAction<string>>;
  value?: string;
  onChange?: (value: string) => void;
  onSubmit?: () => void;
  onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}

const Textarea: FC<TextareaProps> = ({
  rows = 1,
  setModelName,
  value,
  onChange,
  onSubmit,
  onKeyDown,
}) => {
  const { t } = useTranslation();
  const [prompt, setPrompt] = useState<string>('');
  const [isPromptValid, setIsPromptValid] = useState<boolean>(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { isGenerating, selectedConversationId } =
    useAppSelector(conversationSelector) || {};
  const isCurrentConversationGenerating =
    !!isGenerating?.[selectedConversationId || ''];
  const promptValue = value ?? prompt;

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>): void => {
    const value = event.target.value;
    if (onChange) {
      onChange(value);
    } else {
      setPrompt(value);
    }
    setIsPromptValid(value.trim().length > 0);
  };

  const setLLMModel = async () => {
    const response = await fetchModelName();
    if (response.status === 200) {
      setModelName(response.llmModel);
    } else {
      notify(
        response.message || t('llmModelNotSet'),
        NotificationSeverity.ERROR,
      );
    }
  };

  const handleSubmit = () => {
    if (isCurrentConversationGenerating) {
      notify(t('showNotificationWhileStreaming'), NotificationSeverity.WARNING);
      return;
    }

    if (!promptValue.trim()) {
      if (!onChange) {
        setPrompt('');
      }
      setIsPromptValid(false);
      return;
    }

    if (onSubmit) {
      onSubmit();
      return;
    }

    const userPrompt: Message = {
      role: MessageRole.User,
      content: promptValue.trim(),
      time: getCurrentTimeStamp(),
      conversationId: '',
    };

    setLLMModel();

    doConversation({
      conversationId: selectedConversationId || '',
      userPrompt,
    });
    if (!onChange) {
      setPrompt('');
    }
    setIsPromptValid(false);
  };

  const handleKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = (
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (isCurrentConversationGenerating && event.key === 'Enter') {
      event.preventDefault();
      notify(t('showNotificationWhileStreaming'), NotificationSeverity.WARNING);
      return;
    }
    if (!promptValue && event.key === 'Enter') {
      event.preventDefault();
      return;
    }
    if (onKeyDown) {
      onKeyDown(event);
      return;
    }
    if (!event.shiftKey && event.key === 'Enter') {
      event.preventDefault();
      handleSubmit();
    }
  };

  const placeholderText: string = t('askQuestionPlaceholder');

  return (
    <>
      <Group align='flex-end' gap='sm' data-testid='textarea-wrapper' wrap='nowrap'>
        <MantineTextarea
          ref={textareaRef}
          placeholder={placeholderText}
          value={promptValue}
          onKeyDown={handleKeyDown}
          onChange={handleChange}
          autosize
          minRows={rows}
          maxRows={8}
          style={{ flex: 1 }}
          rows={rows}
          data-testid='prompt-textarea'
        />
        <ActionIcon
          aria-label={promptValue.trim() === '' ? t('emptyMessage') : t('submit')}
          onClick={handleSubmit}
          size='lg'
          variant='filled'
          disabled={!isPromptValid || isCurrentConversationGenerating}
          data-testid='submit-prompt'
        >
          <IconArrowRight size={16} />
        </ActionIcon>
      </Group>
    </>
  );
};

export default Textarea;
