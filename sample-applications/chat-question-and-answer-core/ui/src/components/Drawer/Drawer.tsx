// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { FC, ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { Navigation } from '../Conversation/ConversationSideBar.tsx';

interface DrawerProps {
  title?: ReactNode;
  isOpen: boolean;
  close: () => void;
  children?: ReactNode;
}

const Drawer: FC<DrawerProps> = ({ title, isOpen, close, children }) => {
  const { t } = useTranslation();
  return (
    <>
      <div
        onClick={close}
        data-testid='overlay'
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'var(--color-data-source-bg)',
          opacity: isOpen ? '1' : '0',
          visibility: isOpen ? 'visible' : 'hidden',
          transition: 'opacity 0.3s ease-in-out, visibility 0.3s ease-in-out',
          zIndex: 999,
        }}
      />
      <div
        data-testid='drawer-wrapper'
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          height: '100%',
          width: '450px',
          backgroundColor: 'var(--color-white)',
          boxShadow: '-2px 0 5px var(--color-data-source-bs)',
          transform: isOpen ? 'translateX(0)' : 'translateX(100%)',
          transition: 'transform 0.3s ease-in-out',
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Navigation
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <h4>{title || t('drawerTitle')}</h4>
          <button
            onClick={close}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
            }}
          >
            &times;
          </button>
        </Navigation>
        {children}
      </div>
    </>
  );
};

export default Drawer;
