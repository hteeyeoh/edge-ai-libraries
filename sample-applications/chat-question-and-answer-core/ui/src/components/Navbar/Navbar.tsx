// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FC } from 'react';
import { useTranslation } from 'react-i18next';
import styled from 'styled-components';
import { Button } from '@carbon/react';

import { FlexDiv } from '../Conversation/ConversationMessage.tsx';
import { useAppDispatch, useAppSelector } from '../../redux/store.ts';
import {
  conversationSelector,
  newConversation,
} from '../../redux/conversation/conversationSlice';

const StyledDiv = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: var(--color-white);
  background-color: var(--color-info);
  padding: 0.5rem;
  flex-grow: 0;
  position: sticky;
  top: 0;
  z-index: 1;
  height: 3rem;
`;

const Logo = styled.p`
  font-size: 1.4rem;
`;

export const Icon = styled.div`
  margin-right: 0.5rem;
  padding-bottom: 1px;
  font-weight: 100;
  font-size: 1.5rem;
`;

const StyledButton = styled(Button)`
  color: var(--color-white) !important;

  &:disabled {
    filter: grayscale(100%);
    opacity: 0.5;
    cursor: not-allowed;
    background-color: inherit !important;
  }

  &:not(:disabled):hover {
    color: var(--color-button-hover) !important;
    background-color: var(--color-white) !important;
  }

  & > *:not(:last-child) {
    margin-right: 0.5rem;
  }

  display: flex;
  align-items: center;
  --cds-layout-size-height-local: 1.8rem;
  border-radius: 0.5rem;
  font-size: 1rem;
`;

const Navbar: FC = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const { isGenerating, selectedConversationId } =
    useAppSelector(conversationSelector);

  const handleNewConversation = () => {
    dispatch(newConversation());
  };

  return (
    <StyledDiv data-testid='navbar-wrapper'>
      <Logo>{t('chatqna')}</Logo>
      <FlexDiv>
        {selectedConversationId && (
          <StyledButton
            kind='ghost'
            onClick={handleNewConversation}
            disabled={isGenerating}
            data-testid='ask-question-button'
          >
            <Icon style={{ paddingBottom: '1px' }}>+</Icon>
            {t('askQuestion')}
          </StyledButton>
        )}
      </FlexDiv>
    </StyledDiv>
  );
};

export default Navbar;
