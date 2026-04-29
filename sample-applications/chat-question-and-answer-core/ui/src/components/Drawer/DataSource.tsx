// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type FC, type ChangeEvent, useState, useRef } from 'react';
import {
  Box,
  Button,
  Checkbox,
  Drawer,
  Group,
  List,
  Paper,
  Stack,
  Text,
} from '@mantine/core';
import { IconFileText } from '@tabler/icons-react';
import { useTranslation } from 'react-i18next';
import { unwrapResult } from '@reduxjs/toolkit';
import { AxiosError } from 'axios';

import { useAppDispatch, useAppSelector } from '../../redux/store.ts';
import {
  conversationSelector,
  fetchInitialFiles,
  removeAllFiles,
  removeFile,
  uploadFile,
} from '../../redux/conversation/conversationSlice.ts';
import { notify } from '../../components/Notification/notify.ts';
import { NotificationSeverity } from '../../components/Notification/notify.ts';
import { MAX_FILE_SIZE, plainAcceptedFormat } from '../../utils/constant.ts';

interface DataSourceProps {
  buttonDisabled?: boolean;
  close?: () => void;
  opened?: boolean;
  onClose?: () => void;
}

const DataSource: FC<DataSourceProps> = ({ close, opened, onClose }) => {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isValidFile, setIsValidFile] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const drawerOpened = opened ?? true;
  const closeHandler = onClose ?? close ?? (() => undefined);

  const dispatch = useAppDispatch();
  const { files, isUploading, isWaitingForFirstToken } =
    useAppSelector(conversationSelector);
  const isAnyConversationWaiting = Object.values(
    isWaitingForFirstToken || {},
  ).some(Boolean);
  const disableUpload = isUploading || isAnyConversationWaiting;

  const handleFileUpload = async (): Promise<void> => {
    if (file) {
      try {
        closeHandler();
        notify(t('fileUploadStarted'), NotificationSeverity.INFO);
        const response = await dispatch(uploadFile({ file }));
        unwrapResult(response);
        notify(t('fileUploadSuccessful'), NotificationSeverity.SUCCESS);
        dispatch(fetchInitialFiles());
      } catch (error) {
        const axiosError = error as AxiosError;
        notify(`${axiosError.message}`, NotificationSeverity.ERROR);
      } finally {
        setFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const handleSelectFile = (fileName: string): void => {
    setSelectedFiles((prev) =>
      prev.includes(fileName)
        ? prev.filter((name) => name !== fileName)
        : [...prev, fileName],
    );
  };

  const handleDeleteSelected = async (): Promise<void> => {
    if (selectedFiles.length === 0) {
      return;
    }

    const confirmDelete = window.confirm(
      `Delete ${selectedFiles.length} selected file(s)? This action cannot be undone.`,
    );
    if (!confirmDelete) {
      return;
    }

    try {
      for (const fileName of selectedFiles) {
        const response = await dispatch(removeFile({ fileName }));
        unwrapResult(response);
      }
      notify(t('filesSuccessfullyDeleted'), NotificationSeverity.SUCCESS);
      setSelectedFiles([]);
      dispatch(fetchInitialFiles());
    } catch (error) {
      const axiosError = error as AxiosError;
      notify(
        axiosError.message || t('failedToDeleteFiles'),
        NotificationSeverity.ERROR,
      );
    }
  };

  const handleDeleteAll = async (): Promise<void> => {
    if (files.length === 0) {
      return;
    }

    const confirmDelete = window.confirm(
      'Delete all files? This action cannot be undone.',
    );
    if (!confirmDelete) {
      return;
    }

    try {
      const response = await dispatch(removeAllFiles());
      unwrapResult(response);
      notify(t('filesSuccessfullyDeleted'), NotificationSeverity.SUCCESS);
      setSelectedFiles([]);
      dispatch(fetchInitialFiles());
    } catch (error) {
      const axiosError = error as AxiosError;
      notify(
        axiosError.message || t('failedToDeleteFiles'),
        NotificationSeverity.ERROR,
      );
    }
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>): void => {
    event.preventDefault();
    const selectedFile = event.target.files ? event.target.files[0] : null;

    if (selectedFile) {
      const isDuplicate = files.includes(selectedFile.name);
      if (isDuplicate) {
        notify(t('duplicateFileNotification'), NotificationSeverity.WARNING);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        return;
      }
      const fileSizeMB = selectedFile.size / 1024 / 1024;
      const fileNameLower = selectedFile.name.toLowerCase();
      const isSupportedExtension = plainAcceptedFormat.some((extension) =>
        fileNameLower.endsWith(extension),
      );

      if (!isSupportedExtension) {
        notify(t('formatNotification'), NotificationSeverity.ERROR);
        setFile(null);
        setIsValidFile(false);
      } else if (fileSizeMB > MAX_FILE_SIZE) {
        notify(`${t('fileSizeExceeded')}`, NotificationSeverity.WARNING);
        setFile(null);
        setIsValidFile(false);
      } else {
        setFile(selectedFile);
        setIsValidFile(true);
      }
    } else {
      setFile(null);
      setIsValidFile(true);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <Drawer
      opened={drawerOpened}
      onClose={closeHandler}
      title='Data Source'
      position='right'
      data-testid='data-source-wrapper'
    >
      <List size='sm' spacing='xs' mb='md'>
        <List.Item>{t('acceptedFileText')}</List.Item>
        <List.Item>{t('acceptedSizeText')}</List.Item>
      </List>

      <Button onClick={() => fileInputRef.current?.click()} data-testid='add-file-button'>
        {t('addFile')}
      </Button>

      <Box mt='md'>
        <Text fw={600} size='sm' mb='xs'>
          {t('files')}
        </Text>
        {files.length > 0 ? (
          <>
            <Group mb='xs'>
              <Button
                size='xs'
                variant='outline'
                color='red'
                disabled={selectedFiles.length === 0}
                onClick={handleDeleteSelected}
              >
                {t('deleteSelected')}
              </Button>
              <Button
                size='xs'
                variant='outline'
                color='red'
                disabled={files.length === 0}
                onClick={handleDeleteAll}
              >
                {t('deleteAll')}
              </Button>
            </Group>
            <Stack gap='xs' data-testid='existing-files-list'>
              {files.map((existingFile) => (
                <Paper key={existingFile} withBorder p='xs'>
                  <Group gap='xs' wrap='nowrap'>
                    <Checkbox
                      checked={selectedFiles.includes(existingFile)}
                      onChange={() => handleSelectFile(existingFile)}
                      aria-label={`Select ${existingFile}`}
                    />
                    <Text size='sm' style={{ wordBreak: 'break-word' }}>
                      {existingFile}
                    </Text>
                  </Group>
                </Paper>
              ))}
            </Stack>
          </>
        ) : (
          <Text size='sm' c='dimmed'>
            {t('noFilesFound')}
          </Text>
        )}
      </Box>

      {file ? (
        <Paper withBorder p='sm' mt='lg' data-testid='file-container'>
          <Group gap='xs'>
            <IconFileText size={18} />
            <Text size='sm' style={{ wordBreak: 'break-word' }}>
              {file.name}
            </Text>
          </Group>
        </Paper>
      ) : null}

      <input
        ref={fileInputRef}
        type='file'
        accept={plainAcceptedFormat.join(',')}
        style={{ display: 'none' }}
        onChange={handleFileChange}
        data-testid='file-input-field'
      />

      <Box mt='md'>
        {file ? (
          <Button
            disabled={!file || !isValidFile || disableUpload}
            onClick={handleFileUpload}
            data-testid='file-upload-button'
          >
            {t('upload')}
          </Button>
        ) : null}
      </Box>

      {disableUpload ? (
        <List size='sm' mt='md'>
          <List.Item>{t('showNotificationWhileStreaming')}</List.Item>
        </List>
      ) : null}
    </Drawer>
  );
};

export default DataSource;
