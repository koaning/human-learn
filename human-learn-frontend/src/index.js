// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import './styles.css' // <- change './index.css' to './styles.css'
import App from './App';
import * as serviceWorker from './serviceWorker';

ReactDOM.render(<App />, document.getElementById('root'));
serviceWorker.unregister();
