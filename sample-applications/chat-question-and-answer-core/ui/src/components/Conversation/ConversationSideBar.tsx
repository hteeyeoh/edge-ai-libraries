// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  type ButtonHTMLAttributes,
  type FC,
  type HTMLAttributes,
  type PropsWithChildren,
  useState,
} from 'react';
import { ActionIcon, Loader, ScrollAreaAutosize, TextInput, Title } from '@mantine/core';
import { IconCheck, IconEdit, IconTrash, IconX } from '@tabler/icons-react';

import { useAppDispatch, useAppSelector } from '../../redux/store.ts';
import {
  conversationSelector,
  deleteConversation,
  setSelectedConversationId,
  updateConversationTitle,
} from '../../redux/conversation/conversationSlice.ts';
interface ConversationSideBarProps {
  title?: string;
}

export const Navigation: FC<PropsWithChildren<HTMLAttributes<HTMLDivElement>>> = ({
  className,
  children,
  style,
}) => {
  return (
    <div
      className={className}
      style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--mantine-color-gray-3)',
        ...style,
      }}
    >
      {children}
    </div>
  );
};

export const StyledIconButton: FC<
  PropsWithChildren<
    ButtonHTMLAttributes<HTMLButtonElement> & {
      label?: string;
      kind?: string;
      align?: string;
    }
  >
> = ({ children, onClick, ...rest }) => {
  return (
    <button
      type='button'
      onClick={onClick}
      {...rest}
      style={{
        border: 'none',
        background: 'transparent',
        cursor: 'pointer',
        fontSize: '1.1rem',
      }}
    >
      {children}
    </button>
  );
};

const ConversationSideBar: FC<ConversationSideBarProps> = ({ title = 'ChatQnA' }) => {
  const { conversations, selectedConversationId, isGenerating } =
    useAppSelector(conversationSelector);
  const dispatch = useAppDispatch();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>('');
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const handleEditStart = (conversationId: string, currentTitle: string) => {
    setEditingId(conversationId);
    setEditingTitle(currentTitle);
  };

  const handleEditSave = () => {
    if (!editingId) {
      return;
    }
    dispatch(
      updateConversationTitle({
        id: editingId,
        updatedTitle: editingTitle.trim() || 'Untitled',
      }),
    );
    setEditingId(null);
    setEditingTitle('');
  };

  const handleDelete = (conversationId: string) => {
    if (
      window.confirm(
        'Are you sure you want to delete this conversation? This action cannot be undone.',
      )
    ) {
      dispatch(deleteConversation(conversationId));
    }
  };

  const getTitlePreview = (
    input: string,
    maxWords = 3,
    maxCharsNoSpace = 16,
    maxCharsWithSpace = 20,
  ): string => {
    const trimmedInput = input.trim();
    const words = trimmedInput.split(/\s+/).filter(Boolean);
    const hasWhitespace = /\s/.test(trimmedInput);

    // For long no-space titles, fall back to character-based truncation.
    if (!hasWhitespace) {
      return trimmedInput.length > maxCharsNoSpace
        ? `${trimmedInput.slice(0, maxCharsNoSpace)}...`
        : trimmedInput;
    }

    // For titles with spaces, apply word truncation first and then a char cap.
    const wordPreview =
      words.length <= maxWords
        ? trimmedInput
        : `${words.slice(0, maxWords).join(' ')}...`;

    if (wordPreview.length > maxCharsWithSpace) {
      return `${wordPreview.slice(0, maxCharsWithSpace)}...`;
    }
    return wordPreview;
  };

  return (
    <div
      data-testid='conversation-sidebar-wrapper'
      style={{
        width: '280px',
        borderRight: '1px solid var(--mantine-color-gray-3)',
        backgroundColor: 'var(--mantine-color-gray-0)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          backgroundColor: 'var(--mantine-color-blue-6)',
          color: 'var(--color-white)',
          padding: '0 16px',
          height: '64px',
          boxSizing: 'border-box',
          display: 'flex',
          alignItems: 'center',
          borderBottom: '1px solid var(--mantine-color-gray-3)',
        }}
      >
        <Title order={4} m={0} c='white'>
          {title}
        </Title>
      </div>
      <div style={{ padding: '0 16px 8px', fontWeight: 600, fontSize: '0.875rem' }}>
        Chat History
      </div>
      <ScrollAreaAutosize
        type='hover'
        scrollbars='y'
        mah='calc(100vh - 110px)'
        style={{ overflowX: 'hidden' }}
      >
        <div style={{ paddingBottom: 8, paddingRight: 10, maxWidth: '100%' }}>
          {conversations.map((conversation) => {
            const isActive = selectedConversationId === conversation.conversationId;
            return (
              <div
                key={conversation.conversationId}
                onMouseEnter={() => setHoveredId(conversation.conversationId)}
                onMouseLeave={() => setHoveredId(null)}
                onClick={() => {
                  if (editingId !== conversation.conversationId) {
                    dispatch(setSelectedConversationId(conversation.conversationId));
                  }
                }}
                style={{
                  margin: '0 10px 6px',
                  borderRadius: 8,
                  padding: '8px 10px',
                  background: isActive
                    ? 'var(--mantine-color-blue-1)'
                    : 'transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  boxSizing: 'border-box',
                  maxWidth: '100%',
                  gap: 8,
                }}
              >
                {editingId === conversation.conversationId ? (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      flex: 1,
                      gap: 4,
                      minWidth: 0,
                    }}
                  >
                    <TextInput
                      size='xs'
                      value={editingTitle}
                      style={{ flex: 1, minWidth: 0 }}
                      onChange={(e) => setEditingTitle(e.currentTarget.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleEditSave();
                        }
                        if (e.key === 'Escape') {
                          setEditingId(null);
                          setEditingTitle('');
                        }
                      }}
                      autoFocus
                    />
                    <ActionIcon size='sm' color='green' variant='subtle' onClick={handleEditSave}>
                      <IconCheck size={12} />
                    </ActionIcon>
                    <ActionIcon
                      size='sm'
                      color='red'
                      variant='subtle'
                      onClick={() => {
                        setEditingId(null);
                        setEditingTitle('');
                      }}
                    >
                      <IconX size={12} />
                    </ActionIcon>
                  </div>
                ) : (
                  <>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        overflow: 'hidden',
                        flex: 1,
                        minWidth: 0,
                        maxWidth: 'calc(100% - 64px)',
                      }}
                    >
                      <div
                        style={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          minWidth: 0,
                          flex: 1,
                          maxWidth: '100%',
                        }}
                        title={conversation.title || 'Untitled'}
                      >
                        {getTitlePreview(conversation.title || 'Untitled')}
                      </div>
                      {isGenerating?.[conversation.conversationId] ? (
                        <Loader size='xs' color='blue' />
                      ) : null}
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        gap: 4,
                        width: 56,
                        minWidth: 56,
                        justifyContent: 'flex-end',
                        flex: '0 0 56px',
                      }}
                    >
                      <ActionIcon
                        size='sm'
                        variant='light'
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditStart(
                            conversation.conversationId,
                            conversation.title || 'Untitled',
                          );
                        }}
                        style={{
                          minWidth: 24,
                          height: 24,
                          visibility:
                            hoveredId === conversation.conversationId ? 'visible' : 'hidden',
                        }}
                      >
                        <IconEdit size={14} />
                      </ActionIcon>
                      <ActionIcon
                        size='sm'
                        variant='light'
                        color='red'
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(conversation.conversationId);
                        }}
                        style={{
                          minWidth: 24,
                          height: 24,
                          visibility:
                            hoveredId === conversation.conversationId ? 'visible' : 'hidden',
                        }}
                      >
                        <IconTrash size={14} />
                      </ActionIcon>
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>
      </ScrollAreaAutosize>
    </div>
  );
};

export default ConversationSideBar;
