import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Navigate } from "react-router";
import { RouterProvider } from "react-router/dom";
import ChatPage from "@/pages/ChatPage";
import "./index.css";

const router = createBrowserRouter([
    {
        path: "/",
        element: <Navigate to={`/chat/${crypto.randomUUID()}`} replace />,
    },
    {
        path: "/chat/:chatId",
        element: <ChatPage />,
    },
    {
        path: "/chat",
        element: <ChatPage />,
    },
]);

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <RouterProvider router={router} />
    </StrictMode>,
);
