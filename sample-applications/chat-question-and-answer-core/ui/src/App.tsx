// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type FC } from 'react';
import { MantineProvider } from '@mantine/core';
import { Notifications } from '@mantine/notifications';

import NotificationList from './components/Notification/NotificationList.tsx';
import Conversation from './components/Conversation/Conversation.tsx';
import './utils/i18n';
import './App.scss';
import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';

const App: FC = () => {
  return (
    <MantineProvider>
      <Notifications position='top-right' />
      <Conversation title='ChatQnA' />
      <NotificationList />
    </MantineProvider>
  );
};

export default App;
