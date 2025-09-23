/**
 * 主應用程式組件
 *
 * 作者: shaobai
 */

import React, { Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Layout, Spin } from 'antd'
import { LoadingOutlined } from '@ant-design/icons'

import AppHeader from '@components/layout/AppHeader'
import AppFooter from '@components/layout/AppFooter'
import HomePage from '@pages/HomePage'
import PredictionPage from '@pages/PredictionPage'
import LoginPage from '@pages/LoginPage'
import RegisterPage from '@pages/RegisterPage'
import ProfilePage from '@pages/ProfilePage'
import HistoryPage from '@pages/HistoryPage'
import NotFoundPage from '@pages/NotFoundPage'

import '@styles/App.css'

const { Content } = Layout

// 載入動畫
const LoadingSpinner = (
  <div className="loading-container">
    <Spin
      indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />}
      tip="載入中..."
    />
  </div>
)

const App: React.FC = () => {
  return (
    <Layout className="app-layout">
      <AppHeader />

      <Content className="app-content">
        <Suspense fallback={LoadingSpinner}>
          <Routes>
            {/* 公開路由 */}
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />

            {/* 需要認證的路由 */}
            <Route path="/prediction" element={<PredictionPage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/history" element={<HistoryPage />} />

            {/* 404 頁面 */}
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Suspense>
      </Content>

      <AppFooter />
    </Layout>
  )
}

export default App