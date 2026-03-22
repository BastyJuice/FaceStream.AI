import logging
import os
import queue
from server.streaming.video import VideoStreamingServer
from processor.frame import FrameProcessor
from loader.face import FaceLoader
from notification.service import NotificationService
from manager.camera import CameraManager
from config.manager import ConfigManager


# logging.basicConfig(level=logging.DEBUG)


def main():
    camera_manager = None
    frame_processor = None

    try:
        # Konfiguration laden
        config_path = os.path.join('/data', 'config.json')
        config_manager = ConfigManager(config_path)
        config_manager.load_config()

        input_stream_url = config_manager.get('input_stream_url')
        if not input_stream_url:
            raise ValueError("Config value 'input_stream_url' fehlt oder ist leer.")

        # Warteschlangen für Frames
        frame_queue = queue.Queue(maxsize=1)
        processed_frame_queue = queue.Queue(maxsize=1)

        output_size = (
            int(config_manager.get('output_width', 640) or 640),
            int(config_manager.get('output_height', 480) or 480),
        )

        # Kamera-Manager starten
        camera_manager = CameraManager(
            frame_queue,
            input_stream_url,
            output_size,
            config_manager=config_manager,
        )
        camera_manager.start()

        # NotificationService
        notification_service = NotificationService(config_manager)

        # Bekannte Gesichter laden
        face_loader = FaceLoader(config_manager)

        # Frame Processor starten
        frame_processor = FrameProcessor(
            frame_queue,
            processed_frame_queue,
            face_loader,
            config_manager,
            notification_service,
        )
        frame_processor.start()

        # Streaming-Server starten
        server = VideoStreamingServer(config_manager, processed_frame_queue)
        server.run()

    except KeyboardInterrupt:
        logging.info("Shutdown requested by user.")
    except Exception as e:
        logging.exception(f"Fatal error in main: {e}")
        raise
    finally:
        if frame_processor is not None:
            try:
                frame_processor.stop()
                frame_processor.join(timeout=2.0)
            except Exception as e:
                logging.warning(f"Failed to stop FrameProcessor cleanly: {e}")

        if camera_manager is not None:
            try:
                camera_manager.stop()
                camera_manager.join(timeout=2.0)
            except Exception as e:
                logging.warning(f"Failed to stop CameraManager cleanly: {e}")


if __name__ == '__main__':
    main()