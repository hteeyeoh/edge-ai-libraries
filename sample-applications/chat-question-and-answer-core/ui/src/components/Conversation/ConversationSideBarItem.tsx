// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
  useState,
  useRef,
  useEffect,
  type SyntheticEvent,
  type ChangeEvent,
  type KeyboardEvent,
  type FC,
  type ReactNode,
} from 'react';
import { IconEdit, IconTrash } from '@tabler/icons-react';
import { useTranslation } from 'react-i18next';

import { NotificationSeverity, notify } from '../Notification/notify.ts';
import PopupModal from '../PopupModal/PopupModal.tsx';
import {
  deleteConversation,
  updateConversationTitle,
} from '../../redux/conversation/conversationSlice.ts';
import { useAppDispatch } from '../../redux/store.ts';
import Spinner from '../Spinner/Spinner.tsx';

interface ConversationSideBarItemProps {
  title?: string;
  index: string;
  children?: ReactNode;
  isActive?: boolean;
  onClick?: (e: SyntheticEvent) => void;
  showSpinner?: boolean;
}

const ConversationSideBarItem: FC<ConversationSideBarItemProps> = ({
  title,
  index,
  isActive = false,
  onClick,
  showSpinner = false,
}) => {
  const { t } = useTranslation();
  const [isHovered, setIsHovered] = useState<boolean>(false);
  const [showDeleteModal, setShowDeleteModal] = useState<boolean>(false);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editedTitle, setEditedTitle] = useState<string>(title || '');
  const dispatch = useAppDispatch();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDeleteClick = (e: SyntheticEvent) => {
    e.stopPropagation();
    setShowDeleteModal(true);
  };

  const handleDeleteConfirm = async () => {
    setShowDeleteModal(false);
    try {
      dispatch(deleteConversation(index));
      notify(t('conversationDeletionSuccessful'), NotificationSeverity.SUCCESS);
    } catch {
      notify(t('conversationDeletionFailed'), NotificationSeverity.ERROR);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteModal(false);
  };

  const handleEditClick = (e: SyntheticEvent) => {
    e.stopPropagation();
    setIsEditing(true);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setEditedTitle(e.target.value);
  };

  const handleInputBlur = () => {
    setIsEditing(false);
    setEditedTitle(title || t('newChat'));
  };

  const handleInputKeyDown = async (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      if (editedTitle.trim() === '') {
        notify(t('nonEmptyConversationTitle'), NotificationSeverity.WARNING);
        setEditedTitle(title || t('newChat'));
      } else {
        try {
          dispatch(
            updateConversationTitle({ id: index, updatedTitle: editedTitle }),
          );
        } catch {
          notify(
            t('updateConversationTitleFailed'),
            NotificationSeverity.ERROR,
          );
        }
      }
      setIsEditing(false);
    }
  };

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  return (
    <>
      <div
        key={index}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        data-testid='conversation-sidebar-wrapper'
        style={{
          padding: '10px 2px 10px 10px',
          cursor: 'pointer',
          transition: 'background-color 0.3s, color 0.3s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: '40px',
          borderRadius: isActive || isHovered ? '5px' : '0',
          backgroundColor:
            isActive || isHovered ? 'var(--color-active)' : 'transparent',
        }}
      >
        <div
          onClick={onClick}
          style={{
            flexGrow: 1,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
          }}
        >
          {isEditing ? (
            <input
              ref={inputRef}
              value={editedTitle}
              onChange={handleInputChange}
              onBlur={handleInputBlur}
              onKeyDown={handleInputKeyDown}
              style={{
                width: '100%',
                padding: '5px',
                fontSize: '1rem',
                border: '1px solid var(--color-gray-2)',
                borderRadius: '4px',
              }}
            />
          ) : (
            title
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
          {showSpinner ? <Spinner /> : null}
          {(isHovered || isActive) && !isEditing && (
            <>
              <button
                onClick={handleEditClick}
                data-testid='edit-conversation-button'
                style={{
                  borderWidth: 0,
                  background: 'transparent',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <IconEdit size={16} />
              </button>
              <button
                onClick={handleDeleteClick}
                data-testid='delete-conversation-button'
                style={{
                  borderWidth: 0,
                  background: 'transparent',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <IconTrash size={16} />
              </button>
            </>
          )}
        </div>
      </div>

      {showDeleteModal && (
        <PopupModal
          open={showDeleteModal}
          onOpen={setShowDeleteModal}
          headingMsg={t('deleteChat')}
          primaryButtonText={t('delete')}
          secondaryButtonText={t('cancel')}
          onSubmit={handleDeleteConfirm}
          onClose={handleDeleteCancel}
        >
          <p style={{ margin: '1rem' }}>
            {t('thisWillDelete')}
            <strong>{`${title || t('newChat')}`}</strong>.
          </p>
        </PopupModal>
      )}
    </>
  );
};

export default ConversationSideBarItem;
