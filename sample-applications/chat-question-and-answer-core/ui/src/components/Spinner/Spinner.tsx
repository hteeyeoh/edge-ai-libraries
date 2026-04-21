// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { FC } from 'react';

interface SpinnerProps {
  size?: number;
}

const Spinner: FC<SpinnerProps> = ({ size = 50 }) => {
  return (
    <>
      <style>
        {`@keyframes spinnerStretchDelay {
          0%, 40%, 100% { transform: scaleY(0.4); }
          20% { transform: scaleY(1); }
        }`}
      </style>
      <div
        data-testid='spinner-container'
        style={{
          margin: '1rem auto 0',
          display: 'flex',
          justifyContent: 'space-between',
          width: `${size}px`,
          height: `${size}px`,
        }}
      >
        {['-1.2s', '-1.1s', '-1s', '-0.9s', '-0.8s'].map((delay) => (
          <div
            key={delay}
            data-testid='spinner-bar'
            style={{
              backgroundColor: 'var(--color-info)',
              height: '100%',
              width: '2px',
              display: 'inline-block',
              animation: 'spinnerStretchDelay 1.2s infinite ease-in-out',
              animationDelay: delay,
            }}
          />
        ))}
      </div>
    </>
  );
};

export default Spinner;
