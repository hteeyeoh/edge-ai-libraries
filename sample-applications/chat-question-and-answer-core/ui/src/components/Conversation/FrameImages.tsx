// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { FC } from 'react';
import styled from 'styled-components';
import { FrameSource } from '../../redux/conversation/conversation.ts';

interface FrameImagesProps {
  frames: FrameSource[];
}

const FrameContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-top: 1rem;
  width: 100%;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const FrameItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.5rem;
  border: 1px solid var(--color-border, #e0e0e0);
  border-radius: 8px;
  background-color: var(--color-frame-background, #f9f9f9);
`;

const FrameImage = styled.img`
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const FrameCaption = styled.p`
  font-size: 0.875rem;
  color: var(--color-text-secondary, #666);
  margin: 0;
  font-style: italic;
`;

const FrameImages: FC<FrameImagesProps> = ({ frames }) => {
  if (!frames || frames.length === 0) {
    return null;
  }

  return (
    <FrameContainer>
      {frames.map((frame, index) => {
        const { metadata, preview } = frame;
        // Directly use base64 data as image source
        const imageUrl = `data:image/jpeg;base64,${metadata.frame_data}`;

        return (
          <FrameItem key={`frame-${metadata.frame_id}-${index}`}>
            <FrameImage src={imageUrl} alt={preview} />
            {preview && <FrameCaption>{preview}</FrameCaption>}
          </FrameItem>
        );
      })}
    </FrameContainer>
  );
};

export default FrameImages;
