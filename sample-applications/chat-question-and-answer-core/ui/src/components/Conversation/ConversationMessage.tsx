// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { FC } from 'react';
import { Group, Text } from '@mantine/core';
import { IconRobot, IconUser } from '@tabler/icons-react';
import { formatDistanceToNow } from 'date-fns';

interface ConversationMessageProps {
  message: string;
  human: boolean;
  date: number;
  isGenerating?: boolean;
  showBlinkingIndicator?: boolean;
}

export const FlexDiv = ({ children }: { children: React.ReactNode }) => {
  return <Group justify='space-between' align='center'>{children}</Group>;
};

const ConversationMessage: FC<ConversationMessageProps> = ({
  human,
  message,
  date,
  isGenerating = false,
  showBlinkingIndicator = false,
}) => {
  return (
    <div
      style={{
        border: '1px solid var(--mantine-color-gray-3)',
        borderRadius: 10,
        padding: '10px 14px',
        background: human
          ? 'var(--mantine-color-blue-0)'
          : 'var(--mantine-color-gray-0)',
      }}
    >
      <Group justify='space-between' align='flex-start'>
        <Group gap='xs'>
          {human ? <IconUser size={16} /> : <IconRobot size={16} />}
          <Text fw={600} size='sm'>{human ? 'You' : 'Assistant'}</Text>
        </Group>
        <Text size='xs' c='dimmed'>
          {formatDistanceToNow(date, { addSuffix: true })}
        </Text>
      </Group>
      <Text size='sm' mt='xs' style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {message}
        {isGenerating || showBlinkingIndicator ? (
          <span
            data-testid='circle'
            style={{
              marginLeft: 6,
              display: 'inline-block',
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: 'var(--mantine-color-blue-6)',
              animation: 'blink 1s step-end infinite',
            }}
          />
        ) : null}
      </Text>
    </div>
  );
};

export default ConversationMessage;
