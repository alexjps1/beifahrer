import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "Content-Type": "application/json",
    },
});

export interface Message {
    role: "user" | "assistant";
    content: string;
}

export interface ChatRequest {
    user_message_content: string;
}

export interface NewChatResponse {
    chat_id: string;
    message: Message;
}

export interface ChatResponse {
    message: Message;
}

export interface HistoryResponse {
    history: Message[];
}

export async function createChat(
    content: string,
): Promise<NewChatResponse> {
    const response = await apiClient.post<NewChatResponse>("/api/chat/", {
        user_message_content: content,
    });
    return response.data;
}

export async function sendChatMessage(
    chatId: string,
    content: string,
): Promise<ChatResponse> {
    const response = await apiClient.post<ChatResponse>(`/api/chat/${chatId}/`, {
        user_message_content: content,
    });
    return response.data;
}

export async function getChatHistory(chatId: string): Promise<Message[]> {
    const response = await apiClient.get<HistoryResponse>(
        `/api/chat/${chatId}/`,
    );
    return response.data.history;
}

export const isAxiosError = axios.isAxiosError;
