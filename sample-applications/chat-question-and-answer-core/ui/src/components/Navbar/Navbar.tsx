// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type FC, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Button as MantineButton, Group } from '@mantine/core';
import { IconDatabase, IconPlus, IconRss } from '@tabler/icons-react';

import Drawer from '../Drawer/Drawer.tsx';
import { useDisclosure } from '../../hooks/useDisclosure.ts';
import { useAppDispatch, useAppSelector } from '../../redux/store.ts';
import {
    conversationSelector,
    newConversation,
} from '../../redux/conversation/conversationSlice';
import FileList from '../Drawer/FileList.tsx';
import { TitleContainer } from '../Drawer/FileList.tsx';

export const Icon: FC<{ children: ReactNode }> = ({ children }) => (
    <div
        style={{
            marginRight: '0.5rem',
            paddingBottom: '1px',
            fontWeight: 100,
            fontSize: '1.5rem',
            display: 'inline-flex',
            alignItems: 'center',
        }}
    >
        {children}
    </div>
);

const Navbar: FC = () => {
    const { t } = useTranslation();
    const dispatch = useAppDispatch();

    const [isDrawerOpen, { open: openDrawer, close: closeDrawer }] =
        useDisclosure(false);
    const { isGenerating, selectedConversationId } =
        useAppSelector(conversationSelector);
    const isAnyConversationGenerating =
        Object.keys(isGenerating || {}).length > 0;

    const handleNewConversation = () => {
        dispatch(newConversation());
    };

    return (
        <>
            <div
                data-testid='navbar-wrapper'
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    color: 'var(--color-white)',
                    backgroundColor: 'var(--mantine-color-blue-6)',
                    padding: '0.5rem',
                    flexGrow: 0,
                    position: 'sticky',
                    top: 0,
                    zIndex: 1,
                    height: '3rem',
                }}
            >
                <p style={{ fontSize: '1.4rem', margin: 0 }}>{t('chatqna')}</p>
                <Group gap='sm'>
                    {selectedConversationId && (
                        <MantineButton
                            variant='light'
                            color='gray'
                            onClick={handleNewConversation}
                            disabled={isAnyConversationGenerating}
                            data-testid='ask-question-button'
                            style={{ borderRadius: '0.5rem', fontSize: '1rem' }}
                        >
                            <Icon>
                                <IconPlus size={16} />
                            </Icon>
                            {t('askQuestion')}
                        </MantineButton>
                    )}
                    <MantineButton
                        variant='light'
                        color='gray'
                        onClick={openDrawer}
                        disabled={isAnyConversationGenerating}
                        data-testid='manage-context-button'
                        style={{ borderRadius: '0.5rem', fontSize: '1rem' }}
                    >
                        <Icon>
                            <IconDatabase size={16} />
                        </Icon>
                        {t('manageContext')}
                    </MantineButton>
                </Group>
            </div>

            <Drawer
                isOpen={isDrawerOpen}
                close={closeDrawer}
                title={
                    <TitleContainer>
                        <IconRss size={18} style={{ marginRight: '8px' }} />
                        {t('contexts')}
                    </TitleContainer>
                }
            >
                <FileList closeDrawer={closeDrawer} isOpen={isDrawerOpen} />
            </Drawer>
        </>
    );
};

export default Navbar;
