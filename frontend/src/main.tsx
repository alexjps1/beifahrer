import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import Beifahrer from "@/pages/Beifahrer";
import "./index.css";

const router = createBrowserRouter([
    {
        path: "/",
        element: <Beifahrer />,
    },
]);

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <RouterProvider router={router} />
    </StrictMode>,
);
