class DatabaseConnectionError(Exception):
    """데이터베이스 연결 실패 예외"""

    def __init__(self, message: str = "데이터베이스 연결에 실패했습니다", original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self):
        if self.original_error:
            return f"{self.message}: {str(self.original_error)}"
        return self.message


class DatabaseOperationError(Exception):
    """데이터베이스 작업 실패 예외"""

    def __init__(self, operation: str, message: str = None, original_error: Exception = None):
        self.operation = operation
        self.message = message or f"데이터베이스 작업 실패: {operation}"
        self.original_error = original_error
        super().__init__(self.message)


class ServiceUnavailableError(Exception):
    """서비스 사용 불가 예외"""

    def __init__(self, service_name: str, message: str = None):
        self.service_name = service_name
        self.message = message or f"{service_name} 서비스를 사용할 수 없습니다"
        super().__init__(self.message)