// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FC } from 'react';
import { createPortal } from 'react-dom';
import { Button, Group, Modal } from '@mantine/core';
import { useTranslation } from 'react-i18next';

import { PopupModalProps } from './PopupModalProps.ts';

const PopupModal: FC<PopupModalProps> = ({
  open = true,
  passiveModal = false,
  headingMsg,
  primaryButtonText,
  secondaryButtonText,
  size = 'sm',
  children,
  onSubmit,
  onOpen,
  onClose,
  preventCloseOnClickOutside = false,
  primaryButtonDisabled = false,
}) => {
  const { t } = useTranslation();

  return createPortal(
    <Modal
      opened={open}
      onClose={onClose || (() => onOpen(false))}
      title={headingMsg || t('headingMsg')}
      size={size}
      closeOnClickOutside={!preventCloseOnClickOutside}
      data-testid='popup-modal'
    >
      {children}
      {!passiveModal ? (
        <Group justify='flex-end' mt='md'>
          {secondaryButtonText ? (
            <Button variant='default' onClick={onClose || (() => onOpen(false))}>
              {secondaryButtonText}
            </Button>
          ) : null}
          <Button
            onClick={(event) => {
              if (onSubmit) {
                onSubmit(event);
              }
            }}
            disabled={primaryButtonDisabled}
          >
            {primaryButtonText || t('confirm')}
          </Button>
        </Group>
      ) : null}
    </Modal>,
    document.body,
  );
};

export default PopupModal;
