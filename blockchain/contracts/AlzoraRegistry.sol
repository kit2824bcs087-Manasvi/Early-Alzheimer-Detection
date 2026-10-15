// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract AlzoraRegistry {

    struct AnalysisRecord {
        string recordId;
        string mriHash;
        string prediction;
        uint256 confidence;
        string modelVersion;
        uint256 timestamp;
        address submittedBy;
    }

    mapping(string => AnalysisRecord) private records;

    event AnalysisRegistered(
        string indexed recordId,
        string mriHash,
        string prediction,
        uint256 confidence,
        string modelVersion,
        uint256 timestamp,
        address submittedBy
    );

    function registerAnalysis(
        string memory _recordId,
        string memory _mriHash,
        string memory _prediction,
        uint256 _confidence,
        string memory _modelVersion
    ) public {

        require(bytes(_recordId).length > 0, "Record ID required");
        require(bytes(_mriHash).length > 0, "MRI hash required");
        require(bytes(_prediction).length > 0, "Prediction required");

        records[_recordId] = AnalysisRecord({
            recordId: _recordId,
            mriHash: _mriHash,
            prediction: _prediction,
            confidence: _confidence,
            modelVersion: _modelVersion,
            timestamp: block.timestamp,
            submittedBy: msg.sender
        });

        emit AnalysisRegistered(
            _recordId,
            _mriHash,
            _prediction,
            _confidence,
            _modelVersion,
            block.timestamp,
            msg.sender
        );
    }

    function getAnalysis(
        string memory _recordId
    )
        public
        view
        returns (
            string memory,
            string memory,
            string memory,
            uint256,
            string memory,
            uint256,
            address
        )
    {
        AnalysisRecord memory record = records[_recordId];

        return (
            record.recordId,
            record.mriHash,
            record.prediction,
            record.confidence,
            record.modelVersion,
            record.timestamp,
            record.submittedBy
        );
    }

    function verifyMRI(
        string memory _recordId,
        string memory _mriHash
    )
        public
        view
        returns (bool)
    {
        return keccak256(
            bytes(records[_recordId].mriHash)
        ) == keccak256(
            bytes(_mriHash)
        );
    }
}