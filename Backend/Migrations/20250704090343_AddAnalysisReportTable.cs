using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace ESGanalyzer.Backend.Migrations
{
    /// <inheritdoc />
    public partial class AddAnalysisReportTable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Reports");

            migrationBuilder.CreateTable(
                name: "AnalysisReport",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ReportName = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    UserId = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Criteria1 = table.Column<double>(type: "float", nullable: false),
                    Criteria2 = table.Column<double>(type: "float", nullable: false),
                    Criteria3 = table.Column<double>(type: "float", nullable: false),
                    Criteria4 = table.Column<double>(type: "float", nullable: false),
                    Criteria5 = table.Column<double>(type: "float", nullable: false),
                    Criteria6 = table.Column<double>(type: "float", nullable: false),
                    Criteria7 = table.Column<double>(type: "float", nullable: false),
                    Criteria8 = table.Column<double>(type: "float", nullable: false),
                    Criteria9 = table.Column<double>(type: "float", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AnalysisReport", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AnalysisReport");

            migrationBuilder.CreateTable(
                name: "Reports",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    Name = table.Column<string>(type: "nvarchar(max)", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Reports", x => x.Id);
                });
        }
    }
}
