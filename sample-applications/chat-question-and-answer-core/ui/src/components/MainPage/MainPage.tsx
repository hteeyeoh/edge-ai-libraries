// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type FC } from 'react';
import { useTranslation } from 'react-i18next';

import ConversationSideBar from '../Conversation/ConversationSideBar.tsx';
import Conversation from '../Conversation/Conversation.tsx';
import Notice from '../Notice/Notice.tsx';
import Navbar from '../Navbar/Navbar.tsx';

const MainPage: FC = () => {
  const { t } = useTranslation();
  const message = <div>{t('noticeMessage')}</div>;
  const [isNoticeVisible, setIsNoticeVisible] = useState<boolean>(false);

  return (
    <main
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Navbar />
      <button
        data-testid='toggle-notice'
        onClick={() => setIsNoticeVisible(true)}
        style={{ display: 'none' }}
      >
        {t('showNoticeHiddenButton')}
      </button>

      <Notice
        message={message}
        isNoticeVisible={isNoticeVisible}
        setIsNoticeVisible={setIsNoticeVisible}
      />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '15rem auto',
          gridTemplateRows: '1fr',
          paddingInline: 0,
          flexGrow: 1,
          maxInlineSize: '100%',
        }}
      >
        <ConversationSideBar />
        <Conversation />
      </div>
    </main>
  );
};

export default MainPage;
