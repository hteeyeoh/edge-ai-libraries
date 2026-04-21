// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState, type FC, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge, Button, Tabs } from '@mantine/core';
import { IconFileText } from '@tabler/icons-react';

import { useAppSelector } from '../../redux/store.ts';
import { conversationSelector } from '../../redux/conversation/conversationSlice.ts';
import FileLinkManager from './FileLinkManager.tsx';
import DataSource from './DataSource.tsx';

interface FileListProps {
  closeDrawer: () => void;
  isOpen: boolean;
}

export const TitleContainer: FC<{ children: ReactNode }> = ({ children }) => (
  <div style={{ display: 'flex', alignItems: 'center' }}>{children}</div>
);

export const SmallPara: FC<{ children: ReactNode }> = ({ children }) => (
  <p style={{ fontSize: '1rem', marginBottom: '1rem' }}>{children}</p>
);

const FileList: FC<FileListProps> = ({ closeDrawer, isOpen }) => {
  const { t } = useTranslation();
  const { files = [] } = useAppSelector(conversationSelector) || {};
  const [showUploadForm, setShowUploadForm] = useState<boolean>(false);

  const handleButtonClick = () => {
    setShowUploadForm(true);
  };

  useEffect(() => {
    if (!isOpen) {
      setShowUploadForm(false);
    }
  }, [isOpen]);

  return (
    <Tabs defaultValue='files' data-testid='file-list-tabs'>
      <Tabs.List
        style={{
          position: 'sticky',
          top: 0,
          backgroundColor: 'var(--color-white)',
          zIndex: 1,
        }}
      >
        <Tabs.Tab value='files'>
          <div
            style={{
              position: 'relative',
              fontSize: '1.1rem',
              fontWeight: 400,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <IconFileText size={18} style={{ marginRight: '8px' }} />
            {t('files')}
            <Badge
              size='xs'
              variant='filled'
              color='blue'
              style={{
                position: 'absolute',
                top: '-1rem',
                right: '-1.5rem',
                fontSize: '0.6rem',
                minInlineSize: '1rem',
              }}
            >
              {files.length}
            </Badge>
          </div>
        </Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value='files'>
        <div style={{ backgroundColor: 'var(--color-white)' }}>
          {showUploadForm ? (
            <>
              <SmallPara>{t('uploadFileDescription')}</SmallPara>
              <DataSource close={closeDrawer} />
            </>
          ) : (
            <Button
              onClick={handleButtonClick}
              style={{
                minWidth: '100%',
                marginBottom: '1rem',
                fontSize: '1.1rem',
              }}
            >
              +
              {t('addNewFile')}
            </Button>
          )}

          {files.length === 0 ? (
            <div
              style={{
                margin: '1rem auto 0',
                color: 'var(--color-dark-0)',
                lineHeight: 1.3,
              }}
            >
              {t('noFilesFound')}
            </div>
          ) : (
            <FileLinkManager
              showField={showUploadForm}
              closeDrawer={closeDrawer}
            />
          )}
        </div>
      </Tabs.Panel>
    </Tabs>
  );
};

export default FileList;
