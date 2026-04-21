// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState, type FC } from 'react';
import { useTranslation } from 'react-i18next';
import { unwrapResult } from '@reduxjs/toolkit';
import { AxiosError } from 'axios';

import PopupModal from '../PopupModal/PopupModal.tsx';
import { NotificationSeverity, notify } from '../Notification/notify.ts';
import { useAppDispatch, useAppSelector } from '../../redux/store.ts';
import {
  conversationSelector,
  removeAllFiles,
  removeFile,
} from '../../redux/conversation/conversationSlice.ts';

interface FileLinkManagerProps {
  showField: boolean;
  closeDrawer: () => void;
}

const FileLinkManager: FC<FileLinkManagerProps> = ({
  closeDrawer,
  showField,
}) => {
  const { t } = useTranslation();
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [deleteAll, setDeleteAll] = useState<boolean>(false);

  const dispatch = useAppDispatch();
  const { files = [] } = useAppSelector(conversationSelector) || {};

  useEffect(() => {
    setDeleteAll(files.length > 0 && selectedFiles.length === files.length);
  }, [selectedFiles, files]);

  const handleConfirmDelete = async () => {
    setIsModalOpen(false);
    closeDrawer();
    try {
      if (deleteAll) {
        try {
          const response = await dispatch(removeAllFiles());
          unwrapResult(response);
          notify(t('filesSuccessfullyDeleted'), NotificationSeverity.SUCCESS);
        } catch (error) {
          const axiosError = error as AxiosError;
          notify(`${axiosError.message}`, NotificationSeverity.ERROR);
        }
      } else {
        for (const file of selectedFiles) {
          try {
            const response = await dispatch(
              removeFile({
                fileName: file,
              }),
            );
            const result = unwrapResult(response);
            notify(
              `${t('file')} ${result || file} ${t('deletedSuccessfully')}`,
              NotificationSeverity.SUCCESS,
            );
          } catch (error) {
            const axiosError = error as AxiosError;
            notify(`${axiosError.message}`, NotificationSeverity.ERROR);
          }
        }
      }
    } finally {
      setSelectedFiles([]);
      setDeleteAll(false);
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  const handleSelectItem = (file: string) => {
    setSelectedFiles((prevSelectedFiles) =>
      prevSelectedFiles.some((f) => f === file)
        ? prevSelectedFiles.filter((f) => f !== file)
        : [...prevSelectedFiles, file],
    );
  };

  const handleDeleteSelected = () => {
    closeDrawer();
    setIsModalOpen(true);
  };

  const handleDeleteAll = () => {
    closeDrawer();
    setDeleteAll(true);
    setIsModalOpen(true);
  };

  return (
    <>
      <div data-testid='file-link-manager-wrapper' style={{ overflowY: 'auto' }}>
        <p data-testid='files-heading-wrapper' style={{ fontWeight: 500, marginBottom: '0.5rem' }}>
          {t('files')}
        </p>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <button
            onClick={handleDeleteSelected}
            disabled={selectedFiles.length === 0}
            data-testid='handle-delete-selected-button'
            style={{
              backgroundColor: 'var(--color-button)',
              color: 'white',
              border: 'none',
              padding: '5px 10px',
              cursor: selectedFiles.length === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            {t('deleteSelected')}
          </button>
          <button
            onClick={handleDeleteAll}
            disabled={files.length === 0}
            data-testid='handle-delete-all-button'
            style={{
              backgroundColor: 'var(--color-button)',
              color: 'white',
              border: 'none',
              padding: '5px 10px',
              cursor: files.length === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            {t('deleteAll')}
          </button>
        </div>

        <div
          style={{
            overflowY: 'auto',
            maxHeight: showField ? 'max(15vh, 150px)' : 'max(55vh, 475px)',
          }}
        >
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ backgroundColor: 'var(--color-table-head)' }}>
              <tr>
                <th style={{ padding: '10px', border: '1px solid var(--color-gray-4)', textAlign: 'left' }}></th>
                <th style={{ padding: '10px', border: '1px solid var(--color-gray-4)', textAlign: 'left' }}>
                  {t('fileName')}
                </th>
              </tr>
            </thead>

            <tbody>
              {files.map((file, index) => (
                <tr key={index}>
                  <td style={{ padding: '10px', border: '1px solid var(--color-gray-4)' }}>
                    <input
                      type='checkbox'
                      checked={selectedFiles.some((f) => f === file)}
                      onChange={() => handleSelectItem(file)}
                      style={{ margin: 0 }}
                    />
                  </td>
                  <td style={{ padding: '10px', border: '1px solid var(--color-gray-4)', wordBreak: 'break-word', lineHeight: 1.4 }}>
                    {file}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <PopupModal
        open={isModalOpen}
        onOpen={setIsModalOpen}
        onClose={handleCloseModal}
        onSubmit={handleConfirmDelete}
        headingMsg={t('deleteFiles')}
        primaryButtonText={t('confirm')}
        secondaryButtonText={t('cancel')}
      >
        <p>{t('deleteFileDescription')}</p>
        <ul
          style={{
            listStyleType: 'disc',
            paddingLeft: '3rem',
            margin: '0.5rem 0',
            wordBreak: 'break-word',
          }}
        >
          {deleteAll
            ? files.map((file, index) => <li key={index}>{file}</li>)
            : selectedFiles.map((file, index) => <li key={index}>{file}</li>)}
        </ul>
      </PopupModal>
    </>
  );
};

export default FileLinkManager;
