import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { AuthProvider, RequireAuth } from "@/lib/auth";
import DatasetPage from "@/pages/Dataset";
import Home from "@/pages/Home";
import Login from "@/pages/Login";
import ModelPage from "@/pages/Model";
import NotFound from "@/pages/NotFound";
import ProjectPage from "@/pages/Project";
import Users from "@/pages/Users";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route path="login" element={<Login />} />
            <Route index element={<RequireAuth><Home /></RequireAuth>} />
            <Route path="projects/:id" element={<RequireAuth><ProjectPage /></RequireAuth>} />
            <Route path="datasets/:id" element={<RequireAuth><DatasetPage /></RequireAuth>} />
            <Route path="models/:id" element={<RequireAuth><ModelPage /></RequireAuth>} />
            <Route path="admin/users" element={<RequireAuth><Users /></RequireAuth>} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
