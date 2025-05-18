import axios from 'axios';

const client = axios.create({
  baseURL: 'http://localhost:8000', // Update if using Docker or reverse proxy
});

export default client;
