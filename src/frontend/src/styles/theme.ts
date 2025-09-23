/**
 * 中國風主題配置
 *
 * 基於 Ant Design 的算命風格主題
 */

import { ThemeConfig } from 'antd'

export const chineseTheme: ThemeConfig = {
  token: {
    // 主色調 - 金色系
    colorPrimary: '#d4af37', // 金色
    colorSuccess: '#52c41a', // 綠色
    colorWarning: '#faad14', // 橙色
    colorError: '#ff4d4f', // 紅色
    colorInfo: '#1890ff', // 藍色

    // 背景色
    colorBgBase: '#faf8f3', // 米白色背景
    colorBgContainer: '#ffffff',
    colorBgElevated: '#ffffff',

    // 文字色
    colorText: '#2c1810', // 深棕色
    colorTextSecondary: '#8c6239',
    colorTextTertiary: '#a67c52',

    // 邊框
    colorBorder: '#d4af37',
    colorBorderSecondary: '#e6d7b7',

    // 圓角
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,

    // 字體
    fontFamily: '"PingFang SC", "Microsoft YaHei", "SimSun", serif',
    fontSize: 14,
    fontSizeLG: 16,
    fontSizeXL: 20,

    // 陰影
    boxShadow: '0 2px 8px rgba(212, 175, 55, 0.15)',
    boxShadowSecondary: '0 4px 12px rgba(212, 175, 55, 0.1)',

    // 運動效果
    motionDurationSlow: '0.3s',
    motionDurationMid: '0.2s',
    motionDurationFast: '0.1s',
  },
  components: {
    // 按鈕樣式
    Button: {
      primaryShadow: '0 2px 4px rgba(212, 175, 55, 0.3)',
      defaultShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      fontWeight: 600,
    },

    // 卡片樣式
    Card: {
      headerBg: '#f7f2e8',
      boxShadowTertiary: '0 4px 16px rgba(212, 175, 55, 0.1)',
    },

    // 輸入框樣式
    Input: {
      hoverBorderColor: '#d4af37',
      activeBorderColor: '#d4af37',
    },

    // 選單樣式
    Menu: {
      itemBg: 'transparent',
      itemSelectedBg: 'rgba(212, 175, 55, 0.1)',
      itemSelectedColor: '#d4af37',
    },

    // 標籤頁樣式
    Tabs: {
      itemActiveColor: '#d4af37',
      itemSelectedColor: '#d4af37',
      itemHoverColor: '#e6c757',
      inkBarColor: '#d4af37',
    },

    // 表格樣式
    Table: {
      headerBg: '#f7f2e8',
      headerColor: '#2c1810',
      rowHoverBg: 'rgba(212, 175, 55, 0.05)',
    },

    // 模態框樣式
    Modal: {
      headerBg: '#f7f2e8',
      titleColor: '#2c1810',
    },

    // 結果樣式
    Result: {
      titleFontSize: 24,
      iconFontSize: 48,
    },

    // 統計數值樣式
    Statistic: {
      titleFontSize: 14,
      contentFontSize: 24,
      fontFamily: '"PingFang SC", "Microsoft YaHei", "SimSun", serif',
    },

    // 分割線樣式
    Divider: {
      colorSplit: '#e6d7b7',
    },

    // 標籤樣式
    Tag: {
      defaultBg: 'rgba(212, 175, 55, 0.1)',
      defaultColor: '#d4af37',
    },
  },
  algorithm: undefined, // 使用預設演算法
}

// 暗色主題（可選）
export const darkChineseTheme: ThemeConfig = {
  ...chineseTheme,
  token: {
    ...chineseTheme.token,
    colorBgBase: '#1a1a1a',
    colorBgContainer: '#2d2d2d',
    colorBgElevated: '#3d3d3d',
    colorText: '#f0f0f0',
    colorTextSecondary: '#cccccc',
    colorTextTertiary: '#999999',
    colorBorder: '#4d4d4d',
    colorBorderSecondary: '#333333',
  },
}