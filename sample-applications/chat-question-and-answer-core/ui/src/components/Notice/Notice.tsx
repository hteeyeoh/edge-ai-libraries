// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { FC } from 'react';
import { useTranslation } from 'react-i18next';

import { useDisclosure } from '../../hooks/useDisclosure.ts';
import { StyledIconButton } from '../Conversation/ConversationSideBar.tsx';
import { NoticeKind, NoticeProps } from './NoticeProps.ts';

const Notice: FC<NoticeProps> = ({
  message,
  kind = NoticeKind.DEFAULT,
  isNoticeVisible,
  setIsNoticeVisible,
}) => {
  const { t } = useTranslation();
  const [isOpen, { close }] = useDisclosure(true);

  const handleClose = () => {
    close();
    setIsNoticeVisible(false);
  };

  return (
    <>
      {message && isOpen && isNoticeVisible && (
        <div
          data-testid='notice-container'
          style={{
            padding: '0 1rem',
            color: 'var(--color-black)',
            gridColumn: '1 / -1',
            display: 'grid',
            gridTemplateColumns: '1fr auto',
            alignItems: 'center',
            textAlign: 'center',
            backgroundColor: `var(--color-${kind})`,
            transition: 'transform 0.5s ease-in-out, opacity 0.5s ease-in-out',
          }}
        >
          {message}
          <StyledIconButton
            label={t('close')}
            kind='tertiary'
            align='left'
            onClick={handleClose}
            data-testid='close-button'
          >
            &times;
          </StyledIconButton>
        </div>
      )}
    </>
  );
};

export default Notice;
