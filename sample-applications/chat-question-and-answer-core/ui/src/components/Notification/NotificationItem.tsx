// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Alert, CloseButton } from '@mantine/core';
import { type FC, useState, useEffect } from 'react';

import { NotificationItemProps } from './NotificationProps.ts';

const NotificationItem: FC<NotificationItemProps> = ({
  notification,
  onClose,
}) => {
  const { timeout, kind, title } = notification;
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (timeout) {
      const interval = setInterval(() => {
        setProgress((prev) => (prev > 0 ? prev - 100 / (timeout / 100) : 0));
      }, 100);

      return () => clearInterval(interval);
    }
  }, [timeout]);

  // To remove the warning
  useEffect(() => {
    const button = document.querySelector('button[aria-hidden="true"]');
    if (button) {
      button.setAttribute('aria-hidden', 'false');
    }
  }, []);

  const colorByKind: Record<string, string> = {
    error: 'red',
    warning: 'yellow',
    success: 'green',
    info: 'blue',
  };

  return (
    <>
      <style>
        {`@keyframes notificationSlideDown {
          from { top: -50px; opacity: 0; }
          to { top: 5px; opacity: 1; }
        }`}
      </style>
      <div
        data-testid={`notification-item-${notification.id}`}
        style={{
          margin: '0.5rem 0',
          position: 'relative',
          opacity: 1,
          animation: 'notificationSlideDown 0.2s ease-out',
        }}
      >
        <Alert
          color={colorByKind[kind] || 'gray'}
          data-testid='inline-notification'
          variant='filled'
          style={{ color: 'var(--color-white)' }}
        >
          <div style={{ fontWeight: 600, color: 'var(--color-white)' }}>{title}</div>
          <CloseButton
            aria-label='close'
            onClick={onClose}
            c='white'
            style={{ position: 'absolute', top: 8, right: 8 }}
          />
        </Alert>
        {timeout && (
          <div
            data-testid='progress-bar'
            style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              width: '100%',
              height: '2px',
            }}
          >
            <div
              data-testid='progress'
              style={{
                height: '100%',
                transition: 'width 0.1s linear',
                width: `${progress}%`,
                backgroundColor: `var(--color-${kind})`,
              }}
            />
          </div>
        )}
      </div>
    </>
  );
};

export default NotificationItem;
