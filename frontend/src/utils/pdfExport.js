import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';

/**
 * AlphaTab の描画領域を PDF としてエクスポートする
 * @param {HTMLElement} container - キャプチャ対象の DOM 要素 (wrapperRef.current)
 * @param {string} filename - 出力する PDF ファイル名
 */
export const exportToPDF = async (container, filename = 'SoloTab_Export.pdf') => {
    if (!container) throw new Error("Container not found");

    // 一時的にコンテナのスタイルを変更して、スクロールに隠れている部分も含めて全体を描画させる
    const originalHeight = container.style.height;
    const originalOverflow = container.style.overflow;
    const originalMaxHeight = container.style.maxHeight;

    try {
        // html2canvas が全体をキャプチャできるように制限を解除
        container.style.height = 'auto';
        container.style.maxHeight = 'none';
        container.style.overflow = 'visible';

        // 描画の完了を少し待つ
        await new Promise(resolve => setTimeout(resolve, 500));

        const canvas = await html2canvas(container, {
            scale: 2, // 高解像度
            useCORS: true,
            logging: false,
            backgroundColor: '#ffffff'
        });

        const imgData = canvas.toDataURL('image/png');
        
        // A4サイズ (210 x 297 mm)
        const pdf = new jsPDF('p', 'mm', 'a4');
        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = pdf.internal.pageSize.getHeight();

        // キャンバスのサイズを mm に変換 (アスペクト比維持)
        const imgWidth = pdfWidth;
        const imgHeight = (canvas.height * pdfWidth) / canvas.width;

        let heightLeft = imgHeight;
        let position = 0;

        // 1ページ目
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        heightLeft -= pdfHeight;

        // 2ページ目以降
        while (heightLeft > 0) {
            position = heightLeft - imgHeight;
            pdf.addPage();
            pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
            heightLeft -= pdfHeight;
        }

        pdf.save(filename);

    } finally {
        // スタイルを元に戻す
        container.style.height = originalHeight;
        container.style.overflow = originalOverflow;
        container.style.maxHeight = originalMaxHeight;
    }
};
