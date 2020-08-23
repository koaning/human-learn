// src/components/button.js
import React from "react";

export default function Button({ children, ...buttonProps }) {
  return (
    <button
      className="px-2 py-1 bg-gray-100 text-xl shadow-md hover:shadow-lg"
      {...buttonProps}
    >
      {children}
    </button>
  );
}
