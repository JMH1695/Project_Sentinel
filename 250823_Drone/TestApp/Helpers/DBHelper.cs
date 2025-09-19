using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MySql.Data.MySqlClient;

namespace TestApp.Helpers
{
    public static class DBHelper
    {
        // ✅ 입력한 DB 정보 사용
        public static readonly string ConnectionString =
            "Server=localhost;Database=drone_parking;Uid=droneuser;Pwd=Dong@0923;";

        public static MySqlConnection GetConnection()
        {
            return new MySqlConnection(ConnectionString);
        }
    }
}
