import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";
import DatasetPage from "@/pages/Dataset";
import Home from "@/pages/Home";
import ModelPage from "@/pages/Model";
import NotFound from "@/pages/NotFound";
import ProjectPage from "@/pages/Project";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="projects/:id" element={<ProjectPage />} />
          <Route path="datasets/:id" element={<DatasetPage />} />
          <Route path="models/:id" element={<ModelPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
